from __future__ import annotations

from contextlib import contextmanager

import pytest
from sqlalchemy import select, update
from sqlalchemy.engine import Connection
from sqlalchemy.exc import OperationalError

from elspeth.contracts import NodeStateStatus, NodeType, TerminalOutcome, TerminalPath
from elspeth.contracts.audit import TokenRef
from elspeth.contracts.errors import AuditIntegrityError
from elspeth.contracts.schema import SchemaConfig
from elspeth.contracts.schema_contract import SchemaContract
from elspeth.core.landscape import LandscapeDB
from elspeth.core.landscape.errors import LandscapeRecordError
from elspeth.core.landscape.factory import RecorderFactory
from elspeth.core.landscape.schema import node_states_table, token_outcomes_table
from tests.fixtures.landscape import make_factory, make_landscape_db, make_recorder_with_run, register_test_node

_DYNAMIC_SCHEMA = SchemaConfig.from_dict({"mode": "observed"})

# Minimal contract for tests that only care about token lifecycle, not contract content.
# SchemaContract.from_checkpoint validates the hash — this round-trips correctly.
_MINIMAL_CONTRACT = SchemaContract(mode="OBSERVED", fields=(), locked=True)


def _setup(*, run_id: str = "run-1") -> tuple[LandscapeDB, RecorderFactory]:
    setup = make_recorder_with_run(run_id=run_id, source_node_id="source-0", source_plugin_name="csv")
    register_test_node(setup.data_flow, setup.run_id, "agg-0", node_type=NodeType.AGGREGATION, plugin_name="count_agg")
    return setup.db, setup.factory


def _make_batch(factory: RecorderFactory, *, run_id: str = "run-1", batch_id: str = "batch-1") -> str:
    """Helper to create a batch and return its batch_id."""
    batch = factory.execution.create_batch(
        run_id=run_id,
        aggregation_node_id="agg-0",
        batch_id=batch_id,
    )
    return batch.batch_id


def _make_row(factory: RecorderFactory, *, run_id: str = "run-1", row_index: int = 0):
    """Helper to create a row and its initial token."""
    row = factory.data_flow.create_row(
        run_id=run_id,
        source_node_id="source-0",
        row_index=row_index,
        data={"col": f"value-{row_index}"},
        source_row_index=row_index,
        ingest_sequence=row_index,
    )
    token = factory.data_flow.create_token(row.row_id)
    return row, token


def _record_completed_sink_state_with_artifact(
    factory: RecorderFactory,
    *,
    run_id: str,
    token_id: str,
    sink_node_id: str = "sink-0",
) -> str:
    """Create the I1c node-state and artifact witnesses for failsink fallback."""
    register_test_node(
        factory.data_flow,
        run_id,
        sink_node_id,
        node_type=NodeType.SINK,
        plugin_name="failsink",
    )
    state = factory.execution.begin_node_state(
        token_id=token_id,
        node_id=sink_node_id,
        run_id=run_id,
        step_index=0,
        input_data={},
    )
    factory.execution.complete_node_state(
        state_id=state.state_id,
        status=NodeStateStatus.COMPLETED,
        output_data={"written": True},
        duration_ms=1.0,
    )
    artifact = factory.execution.register_artifact(
        run_id=run_id,
        state_id=state.state_id,
        sink_node_id=sink_node_id,
        artifact_type="test",
        path=f"memory://unit/{token_id}",
        content_hash="deadbeef" * 8,
        size_bytes=0,
    )
    return artifact.artifact_id


class TestCreateRow:
    """Tests for DataFlowRepository.create_row."""

    def test_creates_row_with_generated_id(self):
        _db, factory = _setup()
        row = factory.data_flow.create_row(
            run_id="run-1",
            source_node_id="source-0",
            row_index=0,
            data={"name": "Alice"},
            source_row_index=0,
            ingest_sequence=0,
        )
        assert row.row_id is not None
        assert row.run_id == "run-1"
        assert row.source_node_id == "source-0"
        assert row.row_index == 0

    def test_creates_row_with_explicit_id(self):
        _db, factory = _setup()
        row = factory.data_flow.create_row(
            run_id="run-1",
            source_node_id="source-0",
            row_index=0,
            data={"name": "Alice"},
            row_id="custom-row-id",
            source_row_index=0,
            ingest_sequence=0,
        )
        assert row.row_id == "custom-row-id"

    def test_stores_source_data_hash(self):
        _db, factory = _setup()
        row = factory.data_flow.create_row(
            run_id="run-1",
            source_node_id="source-0",
            row_index=0,
            data={"name": "Alice"},
            source_row_index=0,
            ingest_sequence=0,
        )
        assert row.source_data_hash is not None
        assert len(row.source_data_hash) > 0

    def test_deterministic_hash_for_same_data(self):
        _db, factory = _setup()
        row_a = factory.data_flow.create_row(
            run_id="run-1",
            source_node_id="source-0",
            row_index=0,
            data={"name": "Alice"},
            source_row_index=0,
            ingest_sequence=0,
        )
        row_b = factory.data_flow.create_row(
            run_id="run-1",
            source_node_id="source-0",
            row_index=1,
            data={"name": "Alice"},
            source_row_index=1,
            ingest_sequence=1,
        )
        assert row_a.source_data_hash == row_b.source_data_hash

    def test_different_hash_for_different_data(self):
        _db, factory = _setup()
        row_a = factory.data_flow.create_row(
            run_id="run-1",
            source_node_id="source-0",
            row_index=0,
            data={"name": "Alice"},
            source_row_index=0,
            ingest_sequence=0,
        )
        row_b = factory.data_flow.create_row(
            run_id="run-1",
            source_node_id="source-0",
            row_index=1,
            data={"name": "Bob"},
            source_row_index=1,
            ingest_sequence=1,
        )
        assert row_a.source_data_hash != row_b.source_data_hash

    def test_roundtrip_via_get_row(self):
        _db, factory = _setup()
        row = factory.data_flow.create_row(
            run_id="run-1",
            source_node_id="source-0",
            row_index=0,
            data={"name": "Alice"},
            source_row_index=0,
            ingest_sequence=0,
        )
        fetched = factory.query.get_row(row.row_id)
        assert fetched is not None
        assert fetched.row_id == row.row_id
        assert fetched.run_id == row.run_id
        assert fetched.source_node_id == row.source_node_id
        assert fetched.row_index == row.row_index
        assert fetched.source_data_hash == row.source_data_hash

    def test_created_at_is_set(self):
        _db, factory = _setup()
        row = factory.data_flow.create_row(
            run_id="run-1",
            source_node_id="source-0",
            row_index=0,
            data={"col": "val"},
            source_row_index=0,
            ingest_sequence=0,
        )
        assert row.created_at is not None

    def test_multiple_rows_get_unique_ids(self):
        _db, factory = _setup()
        rows = [
            factory.data_flow.create_row(
                run_id="run-1",
                source_node_id="source-0",
                row_index=i,
                data={"i": i},
                source_row_index=i,
                ingest_sequence=i,
            )
            for i in range(5)
        ]
        row_ids = [r.row_id for r in rows]
        assert len(set(row_ids)) == 5


class TestCreateToken:
    """Tests for DataFlowRepository.create_token."""

    def test_creates_token_with_generated_id(self):
        _db, factory = _setup()
        row = factory.data_flow.create_row(
            run_id="run-1",
            source_node_id="source-0",
            row_index=0,
            data={"col": "val"},
            source_row_index=0,
            ingest_sequence=0,
        )
        token = factory.data_flow.create_token(row.row_id)
        assert token.token_id is not None
        assert token.row_id == row.row_id

    def test_creates_token_with_explicit_id(self):
        _db, factory = _setup()
        row = factory.data_flow.create_row(
            run_id="run-1",
            source_node_id="source-0",
            row_index=0,
            data={"col": "val"},
            source_row_index=0,
            ingest_sequence=0,
        )
        token = factory.data_flow.create_token(row.row_id, token_id="custom-token-id")
        assert token.token_id == "custom-token-id"

    def test_creates_token_with_branch_name(self):
        _db, factory = _setup()
        row = factory.data_flow.create_row(
            run_id="run-1",
            source_node_id="source-0",
            row_index=0,
            data={"col": "val"},
            source_row_index=0,
            ingest_sequence=0,
        )
        token = factory.data_flow.create_token(row.row_id, branch_name="path-a", fork_group_id="fg-1")
        assert token.branch_name == "path-a"
        assert token.fork_group_id == "fg-1"

    def test_creates_token_with_fork_group_id(self):
        _db, factory = _setup()
        row = factory.data_flow.create_row(
            run_id="run-1",
            source_node_id="source-0",
            row_index=0,
            data={"col": "val"},
            source_row_index=0,
            ingest_sequence=0,
        )
        token = factory.data_flow.create_token(row.row_id, fork_group_id="fg-1")
        assert token.fork_group_id == "fg-1"

    def test_creates_token_with_join_group_id(self):
        _db, factory = _setup()
        row = factory.data_flow.create_row(
            run_id="run-1",
            source_node_id="source-0",
            row_index=0,
            data={"col": "val"},
            source_row_index=0,
            ingest_sequence=0,
        )
        token = factory.data_flow.create_token(row.row_id, join_group_id="jg-1")
        assert token.join_group_id == "jg-1"

    def test_created_at_is_set(self):
        _db, factory = _setup()
        row = factory.data_flow.create_row(
            run_id="run-1",
            source_node_id="source-0",
            row_index=0,
            data={"col": "val"},
            source_row_index=0,
            ingest_sequence=0,
        )
        token = factory.data_flow.create_token(row.row_id)
        assert token.created_at is not None

    def test_multiple_tokens_for_same_row(self):
        _db, factory = _setup()
        row = factory.data_flow.create_row(
            run_id="run-1",
            source_node_id="source-0",
            row_index=0,
            data={"col": "val"},
            source_row_index=0,
            ingest_sequence=0,
        )
        tokens = [factory.data_flow.create_token(row.row_id) for _ in range(3)]
        token_ids = [t.token_id for t in tokens]
        assert len(set(token_ids)) == 3
        assert all(t.row_id == row.row_id for t in tokens)


class TestForkToken:
    """Tests for DataFlowRepository.fork_token."""

    def test_creates_children_for_each_branch(self):
        _db, factory = _setup()
        row, token = _make_row(factory)
        children, _fork_group_id = factory.data_flow.fork_token(
            parent_ref=TokenRef(token_id=token.token_id, run_id="run-1"),
            row_id=row.row_id,
            branches=["path-a", "path-b", "path-c"],
        )
        assert len(children) == 3
        branch_names = [c.branch_name for c in children]
        assert "path-a" in branch_names
        assert "path-b" in branch_names
        assert "path-c" in branch_names

    def test_children_share_fork_group_id(self):
        _db, factory = _setup()
        row, token = _make_row(factory)
        children, fork_group_id = factory.data_flow.fork_token(
            parent_ref=TokenRef(token_id=token.token_id, run_id="run-1"),
            row_id=row.row_id,
            branches=["path-a", "path-b"],
        )
        assert fork_group_id is not None
        assert all(c.fork_group_id == fork_group_id for c in children)

    def test_children_linked_to_same_row(self):
        _db, factory = _setup()
        row, token = _make_row(factory)
        children, _fg = factory.data_flow.fork_token(
            parent_ref=TokenRef(token_id=token.token_id, run_id="run-1"),
            row_id=row.row_id,
            branches=["path-a", "path-b"],
        )
        assert all(c.row_id == row.row_id for c in children)

    def test_records_parent_forked_outcome(self):
        _db, factory = _setup()
        row, token = _make_row(factory)
        _children, fork_group_id = factory.data_flow.fork_token(
            parent_ref=TokenRef(token_id=token.token_id, run_id="run-1"),
            row_id=row.row_id,
            branches=["path-a", "path-b"],
        )
        outcome = factory.data_flow.get_token_outcome(token.token_id)
        assert outcome is not None
        assert outcome.outcome == TerminalOutcome.TRANSIENT
        assert outcome.path == TerminalPath.FORK_PARENT
        assert outcome.completed is True
        assert outcome.fork_group_id == fork_group_id

    def test_empty_branches_raises_value_error(self):
        _db, factory = _setup()
        row, token = _make_row(factory)
        with pytest.raises(ValueError):
            factory.data_flow.fork_token(
                parent_ref=TokenRef(token_id=token.token_id, run_id="run-1"),
                row_id=row.row_id,
                branches=[],
            )

    def test_children_have_unique_token_ids(self):
        _db, factory = _setup()
        row, token = _make_row(factory)
        children, _fg = factory.data_flow.fork_token(
            parent_ref=TokenRef(token_id=token.token_id, run_id="run-1"),
            row_id=row.row_id,
            branches=["path-a", "path-b", "path-c"],
        )
        token_ids = [c.token_id for c in children]
        assert len(set(token_ids)) == 3

    def test_fork_with_step_in_pipeline(self):
        _db, factory = _setup()
        row, token = _make_row(factory)
        children, _fg = factory.data_flow.fork_token(
            parent_ref=TokenRef(token_id=token.token_id, run_id="run-1"),
            row_id=row.row_id,
            branches=["path-a", "path-b"],
            step_in_pipeline=3,
        )
        assert all(c.step_in_pipeline == 3 for c in children)


class TestCoalesceTokens:
    """Tests for DataFlowRepository.coalesce_tokens."""

    def test_creates_merged_token(self):
        _db, factory = _setup()
        row, token_a = _make_row(factory, row_index=0)
        token_b = factory.data_flow.create_token(row.row_id)
        merged = factory.data_flow.coalesce_tokens(
            parent_refs=[TokenRef(token_id=token_a.token_id, run_id="run-1"), TokenRef(token_id=token_b.token_id, run_id="run-1")],
            row_id=row.row_id,
            merged_payload={"merged": True},
            merged_contract=_MINIMAL_CONTRACT,
        )
        assert merged.token_id is not None
        assert merged.row_id == row.row_id

    def test_merged_token_has_join_group_id(self):
        _db, factory = _setup()
        row, token_a = _make_row(factory, row_index=0)
        token_b = factory.data_flow.create_token(row.row_id)
        merged = factory.data_flow.coalesce_tokens(
            parent_refs=[TokenRef(token_id=token_a.token_id, run_id="run-1"), TokenRef(token_id=token_b.token_id, run_id="run-1")],
            row_id=row.row_id,
            merged_payload={"merged": True},
            merged_contract=_MINIMAL_CONTRACT,
        )
        assert merged.join_group_id is not None

    def test_coalesce_three_tokens(self):
        _db, factory = _setup()
        row, token_a = _make_row(factory, row_index=0)
        token_b = factory.data_flow.create_token(row.row_id)
        token_c = factory.data_flow.create_token(row.row_id)
        merged = factory.data_flow.coalesce_tokens(
            parent_refs=[
                TokenRef(token_id=token_a.token_id, run_id="run-1"),
                TokenRef(token_id=token_b.token_id, run_id="run-1"),
                TokenRef(token_id=token_c.token_id, run_id="run-1"),
            ],
            row_id=row.row_id,
            merged_payload={"merged": True},
            merged_contract=_MINIMAL_CONTRACT,
        )
        assert merged.token_id is not None
        assert merged.join_group_id is not None

    def test_coalesce_with_step_in_pipeline(self):
        _db, factory = _setup()
        row, token_a = _make_row(factory, row_index=0)
        token_b = factory.data_flow.create_token(row.row_id)
        merged = factory.data_flow.coalesce_tokens(
            parent_refs=[TokenRef(token_id=token_a.token_id, run_id="run-1"), TokenRef(token_id=token_b.token_id, run_id="run-1")],
            row_id=row.row_id,
            merged_payload={"merged": True},
            step_in_pipeline=5,
            merged_contract=_MINIMAL_CONTRACT,
        )
        assert merged.step_in_pipeline == 5


class TestExpandToken:
    """Tests for DataFlowRepository.expand_token."""

    def test_creates_n_children(self):
        _db, factory = _setup()
        row, token = _make_row(factory)
        children, expand_group_id = factory.data_flow.expand_token(
            parent_ref=TokenRef(token_id=token.token_id, run_id="run-1"),
            row_id=row.row_id,
            child_payloads=[{"item": i} for i in range(4)],
            output_contract=_MINIMAL_CONTRACT,
        )
        assert len(children) == 4
        assert expand_group_id is not None

    def test_children_share_expand_group_id(self):
        _db, factory = _setup()
        row, token = _make_row(factory)
        children, expand_group_id = factory.data_flow.expand_token(
            parent_ref=TokenRef(token_id=token.token_id, run_id="run-1"),
            row_id=row.row_id,
            child_payloads=[{"item": i} for i in range(3)],
            output_contract=_MINIMAL_CONTRACT,
        )
        assert all(c.expand_group_id == expand_group_id for c in children)

    def test_children_linked_to_same_row(self):
        _db, factory = _setup()
        row, token = _make_row(factory)
        children, _eg = factory.data_flow.expand_token(
            parent_ref=TokenRef(token_id=token.token_id, run_id="run-1"),
            row_id=row.row_id,
            child_payloads=[{"item": 1}, {"item": 2}],
            output_contract=_MINIMAL_CONTRACT,
        )
        assert all(c.row_id == row.row_id for c in children)

    def test_records_parent_expanded_outcome(self):
        _db, factory = _setup()
        row, token = _make_row(factory)
        _children, expand_group_id = factory.data_flow.expand_token(
            parent_ref=TokenRef(token_id=token.token_id, run_id="run-1"),
            row_id=row.row_id,
            child_payloads=[{"item": i} for i in range(3)],
            output_contract=_MINIMAL_CONTRACT,
        )
        outcome = factory.data_flow.get_token_outcome(token.token_id)
        assert outcome is not None
        assert outcome.outcome == TerminalOutcome.TRANSIENT
        assert outcome.path == TerminalPath.EXPAND_PARENT
        assert outcome.completed is True
        assert outcome.expand_group_id == expand_group_id

    def test_count_less_than_one_raises_value_error(self):
        _db, factory = _setup()
        row, token = _make_row(factory)
        with pytest.raises(ValueError):
            factory.data_flow.expand_token(
                parent_ref=TokenRef(token_id=token.token_id, run_id="run-1"),
                row_id=row.row_id,
                child_payloads=[],
                output_contract=_MINIMAL_CONTRACT,
            )

    def test_record_parent_outcome_false_skips_outcome(self):
        _db, factory = _setup()
        row, token = _make_row(factory)
        _children, _eg = factory.data_flow.expand_token(
            parent_ref=TokenRef(token_id=token.token_id, run_id="run-1"),
            row_id=row.row_id,
            child_payloads=[{"item": 1}, {"item": 2}],
            record_parent_outcome=False,
            output_contract=_MINIMAL_CONTRACT,
        )
        outcome = factory.data_flow.get_token_outcome(token.token_id)
        assert outcome is None

    def test_children_have_unique_token_ids(self):
        _db, factory = _setup()
        row, token = _make_row(factory)
        children, _eg = factory.data_flow.expand_token(
            parent_ref=TokenRef(token_id=token.token_id, run_id="run-1"),
            row_id=row.row_id,
            child_payloads=[{"item": i} for i in range(5)],
            output_contract=_MINIMAL_CONTRACT,
        )
        token_ids = [c.token_id for c in children]
        assert len(set(token_ids)) == 5

    def test_expand_with_step_in_pipeline(self):
        _db, factory = _setup()
        row, token = _make_row(factory)
        children, _eg = factory.data_flow.expand_token(
            parent_ref=TokenRef(token_id=token.token_id, run_id="run-1"),
            row_id=row.row_id,
            child_payloads=[{"item": 1}, {"item": 2}],
            step_in_pipeline=7,
            output_contract=_MINIMAL_CONTRACT,
        )
        assert all(c.step_in_pipeline == 7 for c in children)

    def test_expand_count_one(self):
        _db, factory = _setup()
        row, token = _make_row(factory)
        children, expand_group_id = factory.data_flow.expand_token(
            parent_ref=TokenRef(token_id=token.token_id, run_id="run-1"),
            row_id=row.row_id,
            child_payloads=[{"item": 1}],
            output_contract=_MINIMAL_CONTRACT,
        )
        assert len(children) == 1
        assert expand_group_id is not None


class TestValidateOutcomeFields:
    """Tests for outcome field validation via public record_token_outcome API.

    Each outcome variant requires specific companion fields (e.g. COMPLETED
    requires sink_name). These tests verify that record_token_outcome raises
    ValueError when required fields are missing and succeeds when they are
    provided. All assertions go through the public API — no private internals.
    """

    def test_completed_requires_sink_name(self):
        _db, factory = _setup()
        _row, token = _make_row(factory)
        with pytest.raises(ValueError, match="sink_name"):
            factory.data_flow.record_token_outcome(
                ref=TokenRef(token_id=token.token_id, run_id="run-1"),
                outcome=TerminalOutcome.SUCCESS,
                path=TerminalPath.DEFAULT_FLOW,
            )

    def test_completed_accepts_sink_name(self):
        _db, factory = _setup()
        _row, token = _make_row(factory)
        outcome_id = factory.data_flow.record_token_outcome(
            ref=TokenRef(token_id=token.token_id, run_id="run-1"),
            outcome=TerminalOutcome.SUCCESS,
            path=TerminalPath.DEFAULT_FLOW,
            sink_name="output",
        )
        assert outcome_id is not None

    def test_routed_requires_sink_name(self):
        _db, factory = _setup()
        _row, token = _make_row(factory)
        with pytest.raises(ValueError, match="sink_name"):
            factory.data_flow.record_token_outcome(
                ref=TokenRef(token_id=token.token_id, run_id="run-1"),
                outcome=TerminalOutcome.SUCCESS,
                path=TerminalPath.GATE_ROUTED,
            )

    def test_routed_accepts_sink_name(self):
        _db, factory = _setup()
        _row, token = _make_row(factory)
        outcome_id = factory.data_flow.record_token_outcome(
            ref=TokenRef(token_id=token.token_id, run_id="run-1"),
            outcome=TerminalOutcome.SUCCESS,
            path=TerminalPath.GATE_ROUTED,
            sink_name="reject-sink",
        )
        assert outcome_id is not None

    def test_forked_requires_fork_group_id(self):
        _db, factory = _setup()
        _row, token = _make_row(factory)
        with pytest.raises(ValueError, match="fork_group_id"):
            factory.data_flow.record_token_outcome(
                ref=TokenRef(token_id=token.token_id, run_id="run-1"),
                outcome=TerminalOutcome.TRANSIENT,
                path=TerminalPath.FORK_PARENT,
            )

    def test_forked_accepts_fork_group_id(self):
        _db, factory = _setup()
        _row, token = _make_row(factory)
        outcome_id = factory.data_flow.record_token_outcome(
            ref=TokenRef(token_id=token.token_id, run_id="run-1"),
            outcome=TerminalOutcome.TRANSIENT,
            path=TerminalPath.FORK_PARENT,
            fork_group_id="fg-1",
        )
        assert outcome_id is not None

    def test_failed_requires_error_hash(self):
        _db, factory = _setup()
        _row, token = _make_row(factory)
        with pytest.raises(ValueError, match="error_hash"):
            factory.data_flow.record_token_outcome(
                ref=TokenRef(token_id=token.token_id, run_id="run-1"),
                outcome=TerminalOutcome.FAILURE,
                path=TerminalPath.UNROUTED,
            )

    def test_failed_accepts_error_hash(self):
        _db, factory = _setup()
        _row, token = _make_row(factory)
        outcome_id = factory.data_flow.record_token_outcome(
            ref=TokenRef(token_id=token.token_id, run_id="run-1"),
            outcome=TerminalOutcome.FAILURE,
            path=TerminalPath.UNROUTED,
            error_hash="abc123",
        )
        assert outcome_id is not None

    def test_quarantined_requires_error_hash(self):
        _db, factory = _setup()
        _row, token = _make_row(factory)
        with pytest.raises(ValueError, match="error_hash"):
            factory.data_flow.record_token_outcome(
                ref=TokenRef(token_id=token.token_id, run_id="run-1"),
                outcome=TerminalOutcome.FAILURE,
                path=TerminalPath.QUARANTINED_AT_SOURCE,
            )

    def test_quarantined_accepts_error_hash(self):
        _db, factory = _setup()
        _row, token = _make_row(factory)
        outcome_id = factory.data_flow.record_token_outcome(
            ref=TokenRef(token_id=token.token_id, run_id="run-1"),
            outcome=TerminalOutcome.FAILURE,
            path=TerminalPath.QUARANTINED_AT_SOURCE,
            error_hash="abc123",
        )
        assert outcome_id is not None

    def test_quarantined_accepts_error_hash_and_sink_name(self):
        _db, factory = _setup()
        _row, token = _make_row(factory)
        outcome_id = factory.data_flow.record_token_outcome(
            ref=TokenRef(token_id=token.token_id, run_id="run-1"),
            outcome=TerminalOutcome.FAILURE,
            path=TerminalPath.QUARANTINED_AT_SOURCE,
            error_hash="abc123",
            sink_name="quarantine",
        )
        assert outcome_id is not None

    def test_consumed_in_batch_requires_batch_id(self):
        _db, factory = _setup()
        _row, token = _make_row(factory)
        with pytest.raises(ValueError, match="batch_id"):
            factory.data_flow.record_token_outcome(
                ref=TokenRef(token_id=token.token_id, run_id="run-1"),
                outcome=TerminalOutcome.TRANSIENT,
                path=TerminalPath.BATCH_CONSUMED,
            )

    def test_consumed_in_batch_accepts_batch_id(self):
        _db, factory = _setup()
        batch_id = _make_batch(factory)
        _row, token = _make_row(factory)
        outcome_id = factory.data_flow.record_token_outcome(
            ref=TokenRef(token_id=token.token_id, run_id="run-1"),
            outcome=TerminalOutcome.TRANSIENT,
            path=TerminalPath.BATCH_CONSUMED,
            batch_id=batch_id,
        )
        assert outcome_id is not None

    def test_coalesced_requires_join_group_id(self):
        _db, factory = _setup()
        _row, token = _make_row(factory)
        with pytest.raises(ValueError, match="join_group_id"):
            factory.data_flow.record_token_outcome(
                ref=TokenRef(token_id=token.token_id, run_id="run-1"),
                outcome=TerminalOutcome.SUCCESS,
                path=TerminalPath.COALESCED,
                sink_name="output",
            )

    def test_coalesced_accepts_join_group_id(self):
        _db, factory = _setup()
        _row, token = _make_row(factory)
        outcome_id = factory.data_flow.record_token_outcome(
            ref=TokenRef(token_id=token.token_id, run_id="run-1"),
            outcome=TerminalOutcome.SUCCESS,
            path=TerminalPath.COALESCED,
            sink_name="output",
            join_group_id="jg-1",
        )
        assert outcome_id is not None

    def test_coalesced_accepts_join_group_id_without_sink_name(self):
        _db, factory = _setup()
        _row, token = _make_row(factory)
        outcome_id = factory.data_flow.record_token_outcome(
            ref=TokenRef(token_id=token.token_id, run_id="run-1"),
            outcome=TerminalOutcome.SUCCESS,
            path=TerminalPath.COALESCED,
            join_group_id="jg-1",
        )
        assert outcome_id is not None

    def test_expanded_requires_expand_group_id(self):
        _db, factory = _setup()
        _row, token = _make_row(factory)
        with pytest.raises(ValueError, match="expand_group_id"):
            factory.data_flow.record_token_outcome(
                ref=TokenRef(token_id=token.token_id, run_id="run-1"),
                outcome=TerminalOutcome.TRANSIENT,
                path=TerminalPath.EXPAND_PARENT,
            )

    def test_expanded_accepts_expand_group_id(self):
        _db, factory = _setup()
        _row, token = _make_row(factory)
        outcome_id = factory.data_flow.record_token_outcome(
            ref=TokenRef(token_id=token.token_id, run_id="run-1"),
            outcome=TerminalOutcome.TRANSIENT,
            path=TerminalPath.EXPAND_PARENT,
            expand_group_id="eg-1",
        )
        assert outcome_id is not None

    def test_diverted_requires_sink_name(self):
        _db, factory = _setup()
        _row, token = _make_row(factory)
        with pytest.raises(ValueError, match="sink_name"):
            factory.data_flow.record_token_outcome(
                ref=TokenRef(token_id=token.token_id, run_id="run-1"),
                outcome=TerminalOutcome.TRANSIENT,
                path=TerminalPath.SINK_FALLBACK_TO_FAILSINK,
                error_hash="abc123",
            )

    def test_diverted_requires_error_hash(self):
        _db, factory = _setup()
        _row, token = _make_row(factory)
        with pytest.raises(ValueError, match="error_hash"):
            factory.data_flow.record_token_outcome(
                ref=TokenRef(token_id=token.token_id, run_id="run-1"),
                outcome=TerminalOutcome.TRANSIENT,
                path=TerminalPath.SINK_FALLBACK_TO_FAILSINK,
                sink_name="failsink",
            )

    def test_diverted_accepts_sink_name_and_error_hash(self):
        _db, factory = _setup()
        _row, token = _make_row(factory)
        artifact_id = _record_completed_sink_state_with_artifact(
            factory,
            run_id="run-1",
            token_id=token.token_id,
            sink_node_id="sink-0",
        )
        outcome_id = factory.data_flow.record_token_outcome(
            ref=TokenRef(token_id=token.token_id, run_id="run-1"),
            outcome=TerminalOutcome.TRANSIENT,
            path=TerminalPath.SINK_FALLBACK_TO_FAILSINK,
            sink_name="failsink",
            sink_node_id="sink-0",
            artifact_id=artifact_id,
            error_hash="abc123",
        )
        assert outcome_id is not None

    def test_buffered_requires_batch_id(self):
        _db, factory = _setup()
        _row, token = _make_row(factory)
        with pytest.raises(ValueError, match="batch_id"):
            factory.data_flow.record_token_outcome(
                ref=TokenRef(token_id=token.token_id, run_id="run-1"),
                outcome=None,
                path=TerminalPath.BUFFERED,
            )

    def test_buffered_accepts_batch_id(self):
        _db, factory = _setup()
        batch_id = _make_batch(factory)
        _row, token = _make_row(factory)
        outcome_id = factory.data_flow.record_token_outcome(
            ref=TokenRef(token_id=token.token_id, run_id="run-1"),
            outcome=None,
            path=TerminalPath.BUFFERED,
            batch_id=batch_id,
        )
        assert outcome_id is not None


class TestRecordTokenOutcome:
    """Tests for DataFlowRepository.record_token_outcome."""

    def test_records_completed_outcome(self):
        _db, factory = _setup()
        _row, token = _make_row(factory)
        outcome_id = factory.data_flow.record_token_outcome(
            ref=TokenRef(token_id=token.token_id, run_id="run-1"),
            outcome=TerminalOutcome.SUCCESS,
            path=TerminalPath.DEFAULT_FLOW,
            sink_name="output",
        )
        assert outcome_id is not None

    def test_returns_outcome_id_string(self):
        _db, factory = _setup()
        _row, token = _make_row(factory)
        outcome_id = factory.data_flow.record_token_outcome(
            ref=TokenRef(token_id=token.token_id, run_id="run-1"),
            outcome=TerminalOutcome.SUCCESS,
            path=TerminalPath.DEFAULT_FLOW,
            sink_name="output",
        )
        assert isinstance(outcome_id, str)
        assert len(outcome_id) > 0

    def test_roundtrip_via_get_token_outcome(self):
        _db, factory = _setup()
        _row, token = _make_row(factory)
        outcome_id = factory.data_flow.record_token_outcome(
            ref=TokenRef(token_id=token.token_id, run_id="run-1"),
            outcome=TerminalOutcome.SUCCESS,
            path=TerminalPath.DEFAULT_FLOW,
            sink_name="output",
        )
        fetched = factory.data_flow.get_token_outcome(token.token_id)
        assert fetched is not None
        assert fetched.outcome_id == outcome_id
        assert fetched.token_id == token.token_id
        assert fetched.outcome == TerminalOutcome.SUCCESS
        assert fetched.path == TerminalPath.DEFAULT_FLOW
        assert fetched.sink_name == "output"
        assert fetched.completed is True

    def test_records_failed_outcome_with_error_hash(self):
        _db, factory = _setup()
        _row, token = _make_row(factory)
        factory.data_flow.record_token_outcome(
            ref=TokenRef(token_id=token.token_id, run_id="run-1"),
            outcome=TerminalOutcome.FAILURE,
            path=TerminalPath.UNROUTED,
            error_hash="err-hash-abc",
        )
        fetched = factory.data_flow.get_token_outcome(token.token_id)
        assert fetched is not None
        assert fetched.outcome == TerminalOutcome.FAILURE
        assert fetched.path == TerminalPath.UNROUTED
        assert fetched.error_hash == "err-hash-abc"
        assert fetched.completed is True

    def test_records_quarantined_outcome(self):
        _db, factory = _setup()
        _row, token = _make_row(factory)
        factory.data_flow.record_token_outcome(
            ref=TokenRef(token_id=token.token_id, run_id="run-1"),
            outcome=TerminalOutcome.FAILURE,
            path=TerminalPath.QUARANTINED_AT_SOURCE,
            error_hash="quarantine-hash",
        )
        fetched = factory.data_flow.get_token_outcome(token.token_id)
        assert fetched is not None
        assert fetched.outcome == TerminalOutcome.FAILURE
        assert fetched.path == TerminalPath.QUARANTINED_AT_SOURCE
        assert fetched.completed is True

    def test_records_routed_outcome(self):
        _db, factory = _setup()
        _row, token = _make_row(factory)
        factory.data_flow.record_token_outcome(
            ref=TokenRef(token_id=token.token_id, run_id="run-1"),
            outcome=TerminalOutcome.SUCCESS,
            path=TerminalPath.GATE_ROUTED,
            sink_name="reject",
        )
        fetched = factory.data_flow.get_token_outcome(token.token_id)
        assert fetched is not None
        assert fetched.outcome == TerminalOutcome.SUCCESS
        assert fetched.path == TerminalPath.GATE_ROUTED
        assert fetched.sink_name == "reject"

    def test_records_consumed_in_batch_outcome(self):
        _db, factory = _setup()
        batch_id = _make_batch(factory, batch_id="batch-42")
        _row, token = _make_row(factory)
        factory.data_flow.record_token_outcome(
            ref=TokenRef(token_id=token.token_id, run_id="run-1"),
            outcome=TerminalOutcome.TRANSIENT,
            path=TerminalPath.BATCH_CONSUMED,
            batch_id=batch_id,
        )
        fetched = factory.data_flow.get_token_outcome(token.token_id)
        assert fetched is not None
        assert fetched.outcome == TerminalOutcome.TRANSIENT
        assert fetched.path == TerminalPath.BATCH_CONSUMED
        assert fetched.batch_id == "batch-42"

    def test_records_buffered_outcome(self):
        _db, factory = _setup()
        batch_id = _make_batch(factory, batch_id="batch-pending")
        _row, token = _make_row(factory)
        factory.data_flow.record_token_outcome(
            ref=TokenRef(token_id=token.token_id, run_id="run-1"),
            outcome=None,
            path=TerminalPath.BUFFERED,
            batch_id=batch_id,
        )
        fetched = factory.data_flow.get_token_outcome(token.token_id)
        assert fetched is not None
        assert fetched.outcome is None
        assert fetched.path == TerminalPath.BUFFERED
        assert fetched.completed is False

    def test_records_outcome_with_context(self):
        _db, factory = _setup()
        _row, token = _make_row(factory)
        factory.data_flow.record_token_outcome(
            ref=TokenRef(token_id=token.token_id, run_id="run-1"),
            outcome=TerminalOutcome.SUCCESS,
            path=TerminalPath.DEFAULT_FLOW,
            sink_name="output",
            context={"reason": "all good"},
        )
        fetched = factory.data_flow.get_token_outcome(token.token_id)
        assert fetched is not None
        assert fetched.context_json is not None

    def test_recorded_at_is_set(self):
        _db, factory = _setup()
        _row, token = _make_row(factory)
        factory.data_flow.record_token_outcome(
            ref=TokenRef(token_id=token.token_id, run_id="run-1"),
            outcome=TerminalOutcome.SUCCESS,
            path=TerminalPath.DEFAULT_FLOW,
            sink_name="output",
        )
        fetched = factory.data_flow.get_token_outcome(token.token_id)
        assert fetched is not None
        assert fetched.recorded_at is not None


class TestRecordTokenOutcomeAtomicity:
    """Cross-table validation and outcome insertion share one write boundary."""

    def test_repository_owned_call_threads_one_write_connection_through_validation(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _db, factory = _setup()
        _row, token = _make_row(factory)
        outcomes = factory.data_flow.outcomes
        ownership_connections: list[Connection | None] = []
        invariant_connections: list[Connection | None] = []
        original_ownership = outcomes._ownership.validate_token_run_ownership
        original_invariants = outcomes._validate_cross_table_invariants

        def capture_ownership(ref: TokenRef, *, conn: Connection | None = None) -> None:
            ownership_connections.append(conn)
            original_ownership(ref, conn=conn)

        def capture_invariants(
            ref: TokenRef,
            outcome: TerminalOutcome | None,
            path: TerminalPath,
            *,
            sink_name: str | None,
            sink_node_id: str | None,
            artifact_id: str | None,
            conn: Connection | None = None,
            lock_witnesses: bool = True,
        ) -> None:
            invariant_connections.append(conn)
            original_invariants(
                ref,
                outcome,
                path,
                sink_name=sink_name,
                sink_node_id=sink_node_id,
                artifact_id=artifact_id,
                conn=conn,
                lock_witnesses=lock_witnesses,
            )

        monkeypatch.setattr(outcomes._ownership, "validate_token_run_ownership", capture_ownership)
        monkeypatch.setattr(outcomes, "_validate_cross_table_invariants", capture_invariants)

        factory.data_flow.record_token_outcome(
            ref=TokenRef(token_id=token.token_id, run_id="run-1"),
            outcome=TerminalOutcome.SUCCESS,
            path=TerminalPath.DEFAULT_FLOW,
            sink_name="output",
        )

        assert ownership_connections == invariant_connections
        assert len(ownership_connections) == 1
        assert ownership_connections[0] is not None

    def test_injected_failure_rolls_back_validation_side_effect_and_outcome(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        db, factory = _setup()
        _row, token = _make_row(factory)
        register_test_node(factory.data_flow, "run-1", "sink-atomic", node_type=NodeType.SINK, plugin_name="sink")
        state = factory.execution.begin_node_state(
            token_id=token.token_id,
            node_id="sink-atomic",
            run_id="run-1",
            step_index=0,
            input_data={},
        )
        outcomes = factory.data_flow.outcomes
        original_invariants = outcomes._validate_cross_table_invariants

        def fail_after_validation_write(
            ref: TokenRef,
            outcome: TerminalOutcome | None,
            path: TerminalPath,
            *,
            sink_name: str | None,
            sink_node_id: str | None,
            artifact_id: str | None,
            conn: Connection | None = None,
            lock_witnesses: bool = True,
        ) -> None:
            original_invariants(
                ref,
                outcome,
                path,
                sink_name=sink_name,
                sink_node_id=sink_node_id,
                artifact_id=artifact_id,
                conn=conn,
                lock_witnesses=lock_witnesses,
            )
            mutation = (
                update(node_states_table)
                .where(node_states_table.c.state_id == state.state_id)
                .values(status=NodeStateStatus.COMPLETED.value)
            )
            if conn is None:
                # This is the pre-fix shape: validation runs outside the
                # outcome transaction, so its side effect commits independently.
                with db.write_connection() as separate_conn:
                    separate_conn.execute(mutation)
            else:
                conn.execute(mutation)
            raise RuntimeError("injected after cross-table validation")

        monkeypatch.setattr(outcomes, "_validate_cross_table_invariants", fail_after_validation_write)

        with pytest.raises(RuntimeError, match="injected after cross-table validation"):
            factory.data_flow.record_token_outcome(
                ref=TokenRef(token_id=token.token_id, run_id="run-1"),
                outcome=TerminalOutcome.SUCCESS,
                path=TerminalPath.DEFAULT_FLOW,
                sink_name="output",
            )

        with db.read_only_connection() as conn:
            persisted_status = conn.execute(
                select(node_states_table.c.status).where(node_states_table.c.state_id == state.state_id)
            ).scalar_one()
            persisted_outcomes = conn.execute(
                select(token_outcomes_table.c.outcome_id).where(token_outcomes_table.c.token_id == token.token_id)
            ).all()
        assert persisted_status == NodeStateStatus.OPEN.value
        assert persisted_outcomes == []

    def test_caller_supplied_connection_carries_validation_insert_and_outer_rollback(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        db, factory = _setup()
        _row, token = _make_row(factory)
        outcomes = factory.data_flow.outcomes
        seen_connections: list[Connection | None] = []
        original_invariants = outcomes._validate_cross_table_invariants

        def capture_invariants(
            ref: TokenRef,
            outcome: TerminalOutcome | None,
            path: TerminalPath,
            *,
            sink_name: str | None,
            sink_node_id: str | None,
            artifact_id: str | None,
            conn: Connection | None = None,
            lock_witnesses: bool = True,
        ) -> None:
            seen_connections.append(conn)
            original_invariants(
                ref,
                outcome,
                path,
                sink_name=sink_name,
                sink_node_id=sink_node_id,
                artifact_id=artifact_id,
                conn=conn,
                lock_witnesses=lock_witnesses,
            )

        monkeypatch.setattr(outcomes, "_validate_cross_table_invariants", capture_invariants)

        with pytest.raises(RuntimeError, match="outer transaction rollback"), db.write_connection() as caller_conn:
            factory.data_flow.record_token_outcome(
                ref=TokenRef(token_id=token.token_id, run_id="run-1"),
                outcome=TerminalOutcome.SUCCESS,
                path=TerminalPath.DEFAULT_FLOW,
                sink_name="output",
                conn=caller_conn,
            )
            assert seen_connections == [caller_conn]
            raise RuntimeError("outer transaction rollback")

        with db.read_only_connection() as conn:
            persisted_outcomes = conn.execute(
                select(token_outcomes_table.c.outcome_id).where(token_outcomes_table.c.token_id == token.token_id)
            ).all()
        assert persisted_outcomes == []

    def test_context_serialization_failure_opens_no_owned_write_transaction(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        db, factory = _setup()
        _row, token = _make_row(factory)
        opened_transactions: list[bool] = []
        original_write_connection = db.write_connection

        def track_write_connection():
            opened_transactions.append(True)
            return original_write_connection()

        monkeypatch.setattr(db, "write_connection", track_write_connection)

        with pytest.raises(ValueError, match="Cannot canonicalize non-finite float"):
            factory.data_flow.record_token_outcome(
                ref=TokenRef(token_id=token.token_id, run_id="run-1"),
                outcome=TerminalOutcome.SUCCESS,
                path=TerminalPath.DEFAULT_FLOW,
                sink_name="output",
                context={"invalid": float("nan")},
            )

        assert opened_transactions == []
        assert factory.data_flow.get_token_outcome(token.token_id) is None

    def test_context_serialization_failure_precedes_caller_connection_validation(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        db, factory = _setup()
        _row, token = _make_row(factory)
        validation_calls: list[TokenRef] = []
        outcomes = factory.data_flow.outcomes
        original_ownership = outcomes._ownership.validate_token_run_ownership

        def capture_ownership(ref: TokenRef, *, conn: Connection | None = None) -> None:
            validation_calls.append(ref)
            original_ownership(ref, conn=conn)

        monkeypatch.setattr(outcomes._ownership, "validate_token_run_ownership", capture_ownership)

        with db.write_connection() as caller_conn:
            with pytest.raises(ValueError, match="Cannot canonicalize non-finite float"):
                factory.data_flow.record_token_outcome(
                    ref=TokenRef(token_id=token.token_id, run_id="run-1"),
                    outcome=TerminalOutcome.SUCCESS,
                    path=TerminalPath.DEFAULT_FLOW,
                    sink_name="output",
                    context={"invalid": float("nan")},
                    conn=caller_conn,
                )
            persisted_outcomes = caller_conn.execute(
                select(token_outcomes_table.c.outcome_id).where(token_outcomes_table.c.token_id == token.token_id)
            ).all()

        assert validation_calls == []
        assert persisted_outcomes == []

    def test_repository_owned_begin_failure_uses_landscape_error_taxonomy(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        db, factory = _setup()
        _row, token = _make_row(factory)

        @contextmanager
        def fail_begin():
            raise OperationalError("BEGIN IMMEDIATE", {}, RuntimeError("injected begin failure"))
            yield  # pragma: no cover - contextmanager shape only

        monkeypatch.setattr(db, "write_connection", fail_begin)

        with pytest.raises(LandscapeRecordError, match=r"transaction boundary.*OperationalError") as exc_info:
            factory.data_flow.record_token_outcome(
                ref=TokenRef(token_id=token.token_id, run_id="run-1"),
                outcome=TerminalOutcome.SUCCESS,
                path=TerminalPath.DEFAULT_FLOW,
                sink_name="output",
            )

        assert isinstance(exc_info.value.__cause__, OperationalError)
        assert factory.data_flow.get_token_outcome(token.token_id) is None

    def test_repository_owned_commit_failure_rolls_back_and_uses_landscape_error_taxonomy(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        db, factory = _setup()
        _row, token = _make_row(factory)
        original_write_connection = db.write_connection

        @contextmanager
        def fail_commit():
            with original_write_connection() as conn:
                yield conn
                raise OperationalError("COMMIT", {}, RuntimeError("injected commit failure"))

        monkeypatch.setattr(db, "write_connection", fail_commit)

        with pytest.raises(LandscapeRecordError, match=r"transaction boundary.*OperationalError") as exc_info:
            factory.data_flow.record_token_outcome(
                ref=TokenRef(token_id=token.token_id, run_id="run-1"),
                outcome=TerminalOutcome.SUCCESS,
                path=TerminalPath.DEFAULT_FLOW,
                sink_name="output",
            )

        assert isinstance(exc_info.value.__cause__, OperationalError)
        assert factory.data_flow.get_token_outcome(token.token_id) is None

    def test_caller_owned_commit_failure_remains_the_callers_raw_boundary_error(self) -> None:
        db, factory = _setup()
        _row, token = _make_row(factory)

        @contextmanager
        def caller_transaction_with_commit_failure():
            with db.write_connection() as conn:
                yield conn
                raise OperationalError("COMMIT", {}, RuntimeError("caller-owned commit failure"))

        with (
            pytest.raises(OperationalError, match="caller-owned commit failure"),
            caller_transaction_with_commit_failure() as caller_conn,
        ):
            factory.data_flow.record_token_outcome(
                ref=TokenRef(token_id=token.token_id, run_id="run-1"),
                outcome=TerminalOutcome.SUCCESS,
                path=TerminalPath.DEFAULT_FLOW,
                sink_name="output",
                conn=caller_conn,
            )

        assert factory.data_flow.get_token_outcome(token.token_id) is None


class TestGetTokenOutcome:
    """Tests for DataFlowRepository.get_token_outcome."""

    def test_returns_none_for_unknown_token(self):
        _db, factory = _setup()
        result = factory.data_flow.get_token_outcome("nonexistent-token-id")
        assert result is None

    def test_returns_terminal_preferred_over_non_terminal(self):
        _db, factory = _setup()
        batch_id = _make_batch(factory)
        _row, token = _make_row(factory)
        # Record non-terminal first
        factory.data_flow.record_token_outcome(
            ref=TokenRef(token_id=token.token_id, run_id="run-1"),
            outcome=None,
            path=TerminalPath.BUFFERED,
            batch_id=batch_id,
        )
        # Then record terminal
        factory.data_flow.record_token_outcome(
            ref=TokenRef(token_id=token.token_id, run_id="run-1"),
            outcome=TerminalOutcome.SUCCESS,
            path=TerminalPath.DEFAULT_FLOW,
            sink_name="output",
        )
        fetched = factory.data_flow.get_token_outcome(token.token_id)
        assert fetched is not None
        assert fetched.outcome == TerminalOutcome.SUCCESS
        assert fetched.path == TerminalPath.DEFAULT_FLOW
        assert fetched.completed is True

    def test_returns_non_terminal_when_no_terminal_exists(self):
        _db, factory = _setup()
        batch_id = _make_batch(factory)
        _row, token = _make_row(factory)
        factory.data_flow.record_token_outcome(
            ref=TokenRef(token_id=token.token_id, run_id="run-1"),
            outcome=None,
            path=TerminalPath.BUFFERED,
            batch_id=batch_id,
        )
        fetched = factory.data_flow.get_token_outcome(token.token_id)
        assert fetched is not None
        assert fetched.outcome is None
        assert fetched.path == TerminalPath.BUFFERED
        assert fetched.completed is False

    def test_terminal_preferred_regardless_of_insertion_order(self):
        _db, factory = _setup()
        batch_id = _make_batch(factory)
        _row, token = _make_row(factory)
        # Record terminal first
        factory.data_flow.record_token_outcome(
            ref=TokenRef(token_id=token.token_id, run_id="run-1"),
            outcome=TerminalOutcome.SUCCESS,
            path=TerminalPath.DEFAULT_FLOW,
            sink_name="output",
        )
        # Then record non-terminal
        factory.data_flow.record_token_outcome(
            ref=TokenRef(token_id=token.token_id, run_id="run-1"),
            outcome=None,
            path=TerminalPath.BUFFERED,
            batch_id=batch_id,
        )
        fetched = factory.data_flow.get_token_outcome(token.token_id)
        assert fetched is not None
        assert fetched.outcome == TerminalOutcome.SUCCESS
        assert fetched.path == TerminalPath.DEFAULT_FLOW
        assert fetched.completed is True


class TestGetTokenOutcomesForRow:
    """Tests for DataFlowRepository.get_token_outcomes_for_row."""

    def test_returns_empty_list_when_no_outcomes(self):
        _db, factory = _setup()
        row, _token = _make_row(factory)
        outcomes = factory.data_flow.get_token_outcomes_for_row(run_id="run-1", row_id=row.row_id)
        assert outcomes == []

    def test_returns_all_outcomes_for_row(self):
        _db, factory = _setup()
        row, token_a = _make_row(factory, row_index=0)
        token_b = factory.data_flow.create_token(row.row_id)
        factory.data_flow.record_token_outcome(
            ref=TokenRef(token_id=token_a.token_id, run_id="run-1"),
            outcome=TerminalOutcome.SUCCESS,
            path=TerminalPath.DEFAULT_FLOW,
            sink_name="output",
        )
        factory.data_flow.record_token_outcome(
            ref=TokenRef(token_id=token_b.token_id, run_id="run-1"),
            outcome=TerminalOutcome.SUCCESS,
            path=TerminalPath.GATE_ROUTED,
            sink_name="reject",
        )
        outcomes = factory.data_flow.get_token_outcomes_for_row(run_id="run-1", row_id=row.row_id)
        assert len(outcomes) == 2
        pairs = {(o.outcome, o.path) for o in outcomes}
        assert (TerminalOutcome.SUCCESS, TerminalPath.DEFAULT_FLOW) in pairs
        assert (TerminalOutcome.SUCCESS, TerminalPath.GATE_ROUTED) in pairs

    def test_does_not_return_outcomes_from_other_rows(self):
        _db, factory = _setup()
        row_a, token_a = _make_row(factory, row_index=0)
        _row_b, token_b = _make_row(factory, row_index=1)
        factory.data_flow.record_token_outcome(
            ref=TokenRef(token_id=token_a.token_id, run_id="run-1"),
            outcome=TerminalOutcome.SUCCESS,
            path=TerminalPath.DEFAULT_FLOW,
            sink_name="output",
        )
        factory.data_flow.record_token_outcome(
            ref=TokenRef(token_id=token_b.token_id, run_id="run-1"),
            outcome=TerminalOutcome.FAILURE,
            path=TerminalPath.UNROUTED,
            error_hash="err-hash",
        )
        outcomes_a = factory.data_flow.get_token_outcomes_for_row(run_id="run-1", row_id=row_a.row_id)
        assert len(outcomes_a) == 1
        assert outcomes_a[0].outcome == TerminalOutcome.SUCCESS
        assert outcomes_a[0].path == TerminalPath.DEFAULT_FLOW

    def test_returns_empty_for_nonexistent_row(self):
        _db, factory = _setup()
        outcomes = factory.data_flow.get_token_outcomes_for_row(run_id="run-1", row_id="no-such-row")
        assert outcomes == []

    def test_returns_multiple_outcomes_per_token(self):
        _db, factory = _setup()
        batch_id = _make_batch(factory)
        row, token = _make_row(factory, row_index=0)
        factory.data_flow.record_token_outcome(
            ref=TokenRef(token_id=token.token_id, run_id="run-1"),
            outcome=None,
            path=TerminalPath.BUFFERED,
            batch_id=batch_id,
        )
        factory.data_flow.record_token_outcome(
            ref=TokenRef(token_id=token.token_id, run_id="run-1"),
            outcome=TerminalOutcome.SUCCESS,
            path=TerminalPath.DEFAULT_FLOW,
            sink_name="output",
        )
        outcomes = factory.data_flow.get_token_outcomes_for_row(run_id="run-1", row_id=row.row_id)
        assert len(outcomes) == 2

    def test_does_not_return_outcomes_from_other_runs(self):
        _db_a, factory_a = _setup(run_id="run-A")
        row_a, token_a = _make_row(factory_a, run_id="run-A", row_index=0)
        factory_a.data_flow.record_token_outcome(
            ref=TokenRef(token_id=token_a.token_id, run_id="run-A"),
            outcome=TerminalOutcome.SUCCESS,
            path=TerminalPath.DEFAULT_FLOW,
            sink_name="output",
        )
        # Query with a different run_id
        outcomes = factory_a.data_flow.get_token_outcomes_for_row(run_id="run-B", row_id=row_a.row_id)
        assert outcomes == []


# ===========================================================================
# Regression tests: P1-2026-02-14 cross-run contamination prevention
# ===========================================================================


def _setup_two_runs() -> tuple[LandscapeDB, RecorderFactory]:
    """Set up a shared database with two runs, each with a source and aggregation node."""
    db = make_landscape_db()
    factory = make_factory(db)

    # Run A
    factory.run_lifecycle.begin_run(config={}, canonical_version="v1", run_id="run-A")
    factory.data_flow.register_node(
        run_id="run-A",
        plugin_name="csv",
        node_type=NodeType.SOURCE,
        plugin_version="1.0",
        config={},
        node_id="source-0",
        schema_config=_DYNAMIC_SCHEMA,
    )
    factory.data_flow.register_node(
        run_id="run-A",
        plugin_name="count_agg",
        node_type=NodeType.AGGREGATION,
        plugin_version="1.0",
        config={},
        node_id="agg-0",
        schema_config=_DYNAMIC_SCHEMA,
    )

    # Run B
    factory.run_lifecycle.begin_run(config={}, canonical_version="v1", run_id="run-B")
    factory.data_flow.register_node(
        run_id="run-B",
        plugin_name="csv",
        node_type=NodeType.SOURCE,
        plugin_version="1.0",
        config={},
        node_id="source-0",
        schema_config=_DYNAMIC_SCHEMA,
    )
    factory.data_flow.register_node(
        run_id="run-B",
        plugin_name="count_agg",
        node_type=NodeType.AGGREGATION,
        plugin_version="1.0",
        config={},
        node_id="agg-0",
        schema_config=_DYNAMIC_SCHEMA,
    )

    return db, factory


class TestCrossRunContaminationPrevention:
    """P1-2026-02-14: Token lifecycle methods must crash on cross-run contamination.

    These tests verify that recording audit records under the wrong run_id
    raises AuditIntegrityError immediately, rather than silently corrupting
    the audit trail.
    """

    def test_record_token_outcome_rejects_wrong_run_id(self):
        """record_token_outcome must crash if token belongs to a different run."""
        _db, factory = _setup_two_runs()

        # Create row and token in run-A
        row_a = factory.data_flow.create_row(
            run_id="run-A",
            source_node_id="source-0",
            row_index=0,
            data={"col": "value"},
            source_row_index=0,
            ingest_sequence=0,
        )
        token_a = factory.data_flow.create_token(row_a.row_id)

        # Attempt to record outcome under run-B -- must crash
        with pytest.raises(AuditIntegrityError, match="Cross-run contamination"):
            factory.data_flow.record_token_outcome(
                ref=TokenRef(token_id=token_a.token_id, run_id="run-B"),
                outcome=TerminalOutcome.SUCCESS,
                path=TerminalPath.DEFAULT_FLOW,
                sink_name="output",
            )

    def test_record_token_outcome_accepts_correct_run_id(self):
        """record_token_outcome must succeed when run_id matches token ownership."""
        _db, factory = _setup_two_runs()

        row_a = factory.data_flow.create_row(
            run_id="run-A",
            source_node_id="source-0",
            row_index=0,
            data={"col": "value"},
            source_row_index=0,
            ingest_sequence=0,
        )
        token_a = factory.data_flow.create_token(row_a.row_id)

        # Recording with the correct run_id should succeed
        outcome_id = factory.data_flow.record_token_outcome(
            ref=TokenRef(token_id=token_a.token_id, run_id="run-A"),
            outcome=TerminalOutcome.SUCCESS,
            path=TerminalPath.DEFAULT_FLOW,
            sink_name="output",
        )
        assert outcome_id is not None

    def test_fork_token_rejects_wrong_run_id(self):
        """fork_token must crash if parent token belongs to a different run."""
        _db, factory = _setup_two_runs()

        row_a = factory.data_flow.create_row(
            run_id="run-A",
            source_node_id="source-0",
            row_index=0,
            data={"col": "value"},
            source_row_index=0,
            ingest_sequence=0,
        )
        token_a = factory.data_flow.create_token(row_a.row_id)

        with pytest.raises(AuditIntegrityError, match="Cross-run contamination"):
            factory.data_flow.fork_token(
                parent_ref=TokenRef(token_id=token_a.token_id, run_id="run-B"),
                row_id=row_a.row_id,
                branches=["path-a", "path-b"],
            )

    def test_fork_token_rejects_wrong_row_id(self):
        """fork_token must crash if parent token belongs to a different row."""
        _db, factory = _setup_two_runs()

        row_a = factory.data_flow.create_row(
            run_id="run-A",
            source_node_id="source-0",
            row_index=0,
            data={"col": "value-a"},
            source_row_index=0,
            ingest_sequence=0,
        )
        row_b = factory.data_flow.create_row(
            run_id="run-A",
            source_node_id="source-0",
            row_index=1,
            data={"col": "value-b"},
            source_row_index=1,
            ingest_sequence=1,
        )
        token_a = factory.data_flow.create_token(row_a.row_id)

        with pytest.raises(AuditIntegrityError, match="Cross-row lineage"):
            factory.data_flow.fork_token(
                parent_ref=TokenRef(token_id=token_a.token_id, run_id="run-A"),
                row_id=row_b.row_id,
                branches=["path-a"],
            )

    def test_fork_token_accepts_correct_ownership(self):
        """fork_token must succeed when run_id and row_id match parent token."""
        _db, factory = _setup_two_runs()

        row_a = factory.data_flow.create_row(
            run_id="run-A",
            source_node_id="source-0",
            row_index=0,
            data={"col": "value"},
            source_row_index=0,
            ingest_sequence=0,
        )
        token_a = factory.data_flow.create_token(row_a.row_id)

        children, fg = factory.data_flow.fork_token(
            parent_ref=TokenRef(token_id=token_a.token_id, run_id="run-A"),
            row_id=row_a.row_id,
            branches=["path-a", "path-b"],
        )
        assert len(children) == 2
        assert fg is not None

    def test_expand_token_rejects_wrong_run_id(self):
        """expand_token must crash if parent token belongs to a different run."""
        _db, factory = _setup_two_runs()

        row_a = factory.data_flow.create_row(
            run_id="run-A",
            source_node_id="source-0",
            row_index=0,
            data={"col": "value"},
            source_row_index=0,
            ingest_sequence=0,
        )
        token_a = factory.data_flow.create_token(row_a.row_id)

        with pytest.raises(AuditIntegrityError, match="Cross-run contamination"):
            factory.data_flow.expand_token(
                parent_ref=TokenRef(token_id=token_a.token_id, run_id="run-B"),
                row_id=row_a.row_id,
                child_payloads=[{"item": i} for i in range(3)],
                output_contract=_MINIMAL_CONTRACT,
            )

    def test_expand_token_rejects_wrong_row_id(self):
        """expand_token must crash if parent token belongs to a different row."""
        _db, factory = _setup_two_runs()

        row_a = factory.data_flow.create_row(
            run_id="run-A",
            source_node_id="source-0",
            row_index=0,
            data={"col": "value-a"},
            source_row_index=0,
            ingest_sequence=0,
        )
        row_b = factory.data_flow.create_row(
            run_id="run-A",
            source_node_id="source-0",
            row_index=1,
            data={"col": "value-b"},
            source_row_index=1,
            ingest_sequence=1,
        )
        token_a = factory.data_flow.create_token(row_a.row_id)

        with pytest.raises(AuditIntegrityError, match="Cross-row lineage"):
            factory.data_flow.expand_token(
                parent_ref=TokenRef(token_id=token_a.token_id, run_id="run-A"),
                row_id=row_b.row_id,
                child_payloads=[{"item": 1}, {"item": 2}],
                output_contract=_MINIMAL_CONTRACT,
            )

    def test_expand_token_accepts_correct_ownership(self):
        """expand_token must succeed when run_id and row_id match parent token."""
        _db, factory = _setup_two_runs()

        row_a = factory.data_flow.create_row(
            run_id="run-A",
            source_node_id="source-0",
            row_index=0,
            data={"col": "value"},
            source_row_index=0,
            ingest_sequence=0,
        )
        token_a = factory.data_flow.create_token(row_a.row_id)

        children, eg = factory.data_flow.expand_token(
            parent_ref=TokenRef(token_id=token_a.token_id, run_id="run-A"),
            row_id=row_a.row_id,
            child_payloads=[{"item": i} for i in range(3)],
            output_contract=_MINIMAL_CONTRACT,
        )
        assert len(children) == 3
        assert eg is not None

    def test_coalesce_tokens_rejects_cross_run_parents(self):
        """coalesce_tokens must crash if parent tokens belong to different runs."""
        _db, factory = _setup_two_runs()

        row_a = factory.data_flow.create_row(
            run_id="run-A",
            source_node_id="source-0",
            row_index=0,
            data={"col": "value-a"},
            source_row_index=0,
            ingest_sequence=0,
        )
        token_a = factory.data_flow.create_token(row_a.row_id)

        row_b = factory.data_flow.create_row(
            run_id="run-B",
            source_node_id="source-0",
            row_index=0,
            data={"col": "value-b"},
            source_row_index=0,
            ingest_sequence=0,
        )
        token_b = factory.data_flow.create_token(row_b.row_id)

        # token_a belongs to run-A, token_b belongs to run-B
        # coalesce requires row_id match, so this will fail on row ownership first
        with pytest.raises(AuditIntegrityError):
            factory.data_flow.coalesce_tokens(
                parent_refs=[TokenRef(token_id=token_a.token_id, run_id="run-A"), TokenRef(token_id=token_b.token_id, run_id="run-B")],
                row_id=row_a.row_id,
                merged_payload={"merged": True},
                merged_contract=_MINIMAL_CONTRACT,
            )

    def test_coalesce_tokens_rejects_wrong_row_id(self):
        """coalesce_tokens must crash if parent token belongs to a different row."""
        _db, factory = _setup_two_runs()

        row_a = factory.data_flow.create_row(
            run_id="run-A",
            source_node_id="source-0",
            row_index=0,
            data={"col": "value-a"},
            source_row_index=0,
            ingest_sequence=0,
        )
        row_b = factory.data_flow.create_row(
            run_id="run-A",
            source_node_id="source-0",
            row_index=1,
            data={"col": "value-b"},
            source_row_index=1,
            ingest_sequence=1,
        )
        token_a = factory.data_flow.create_token(row_a.row_id)
        token_b = factory.data_flow.create_token(row_a.row_id)

        # Both tokens belong to row_a, but we say row_b
        with pytest.raises(AuditIntegrityError, match="Cross-row lineage"):
            factory.data_flow.coalesce_tokens(
                parent_refs=[TokenRef(token_id=token_a.token_id, run_id="run-A"), TokenRef(token_id=token_b.token_id, run_id="run-A")],
                row_id=row_b.row_id,
                merged_payload={"merged": True},
                merged_contract=_MINIMAL_CONTRACT,
            )

    def test_coalesce_tokens_accepts_correct_ownership(self):
        """coalesce_tokens must succeed when all parents belong to the same row/run."""
        _db, factory = _setup_two_runs()

        row_a = factory.data_flow.create_row(
            run_id="run-A",
            source_node_id="source-0",
            row_index=0,
            data={"col": "value"},
            source_row_index=0,
            ingest_sequence=0,
        )
        token_a = factory.data_flow.create_token(row_a.row_id)
        token_b = factory.data_flow.create_token(row_a.row_id)

        merged = factory.data_flow.coalesce_tokens(
            parent_refs=[TokenRef(token_id=token_a.token_id, run_id="run-A"), TokenRef(token_id=token_b.token_id, run_id="run-A")],
            row_id=row_a.row_id,
            merged_payload={"merged": True},
            merged_contract=_MINIMAL_CONTRACT,
        )
        assert merged.token_id is not None
        assert merged.run_id == "run-A"


class TestTokenRunIdConsistency:
    """P1-2026-02-14: Tokens must store run_id and derive it from their row.

    These tests verify that create_token correctly derives run_id from the
    row record and stores it, ensuring schema-level enforcement via composite FKs.
    """

    def test_create_token_stores_run_id(self):
        """create_token must derive and store run_id from the row's run."""
        _db, factory = _setup(run_id="run-1")
        row = factory.data_flow.create_row(
            run_id="run-1",
            source_node_id="source-0",
            row_index=0,
            data={"col": "value"},
            source_row_index=0,
            ingest_sequence=0,
        )
        token = factory.data_flow.create_token(row.row_id)
        assert token.run_id == "run-1"

    def test_create_token_for_nonexistent_row_crashes(self):
        """create_token must crash if the row_id does not exist (Tier 1 violation)."""
        _db, factory = _setup(run_id="run-1")
        with pytest.raises(AuditIntegrityError, match="does not exist"):
            factory.data_flow.create_token("nonexistent-row-id")

    def test_fork_children_have_run_id(self):
        """Forked child tokens must inherit run_id from parent."""
        _db, factory = _setup(run_id="run-1")
        row, token = _make_row(factory)
        children, _fg = factory.data_flow.fork_token(
            parent_ref=TokenRef(token_id=token.token_id, run_id="run-1"),
            row_id=row.row_id,
            branches=["path-a", "path-b"],
        )
        assert all(c.run_id == "run-1" for c in children)

    def test_expand_children_have_run_id(self):
        """Expanded child tokens must inherit run_id from parent."""
        _db, factory = _setup(run_id="run-1")
        row, token = _make_row(factory)
        children, _eg = factory.data_flow.expand_token(
            parent_ref=TokenRef(token_id=token.token_id, run_id="run-1"),
            row_id=row.row_id,
            child_payloads=[{"item": i} for i in range(3)],
            output_contract=_MINIMAL_CONTRACT,
        )
        assert all(c.run_id == "run-1" for c in children)

    def test_coalesced_token_has_run_id(self):
        """Coalesced token must inherit run_id from parents."""
        _db, factory = _setup(run_id="run-1")
        row, token_a = _make_row(factory, row_index=0)
        token_b = factory.data_flow.create_token(row.row_id)
        merged = factory.data_flow.coalesce_tokens(
            parent_refs=[TokenRef(token_id=token_a.token_id, run_id="run-1"), TokenRef(token_id=token_b.token_id, run_id="run-1")],
            row_id=row.row_id,
            merged_payload={"merged": True},
            merged_contract=_MINIMAL_CONTRACT,
        )
        assert merged.run_id == "run-1"

    def test_token_roundtrip_preserves_run_id(self):
        """Token run_id should survive DB roundtrip via get_token."""
        _db, factory = _setup(run_id="run-1")
        row = factory.data_flow.create_row(
            run_id="run-1",
            source_node_id="source-0",
            row_index=0,
            data={"col": "value"},
            source_row_index=0,
            ingest_sequence=0,
        )
        token = factory.data_flow.create_token(row.row_id)
        fetched = factory.query.get_token(token.token_id)
        assert fetched is not None
        assert fetched.run_id == "run-1"

    def test_schema_composite_fk_prevents_cross_run_outcome(self):
        """Schema composite FK on token_outcomes must reject mismatched (token_id, run_id).

        Even if the application-level check were bypassed, the database constraint
        should reject the insert.
        """
        from sqlalchemy.exc import IntegrityError

        from elspeth.core.landscape._helpers import generate_id, now
        from elspeth.core.landscape.schema import token_outcomes_table

        db = make_landscape_db()
        factory = make_factory(db)

        # Set up run-A with row + token
        factory.run_lifecycle.begin_run(config={}, canonical_version="v1", run_id="run-A")
        factory.data_flow.register_node(
            run_id="run-A",
            plugin_name="csv",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            node_id="source-0",
            schema_config=_DYNAMIC_SCHEMA,
        )
        row_a = factory.data_flow.create_row(
            run_id="run-A",
            source_node_id="source-0",
            row_index=0,
            data={"col": "value"},
            source_row_index=0,
            ingest_sequence=0,
        )
        token_a = factory.data_flow.create_token(row_a.row_id)

        # Set up run-B (but don't create any tokens in it)
        factory.run_lifecycle.begin_run(config={}, canonical_version="v1", run_id="run-B")
        factory.data_flow.register_node(
            run_id="run-B",
            plugin_name="csv",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            node_id="source-0",
            schema_config=_DYNAMIC_SCHEMA,
        )

        # Try to insert directly into token_outcomes with mismatched (token_id, run_id)
        # token_a belongs to run-A, but we try to record under run-B
        # The composite FK should reject this
        with pytest.raises(IntegrityError), db.connection() as conn:
            conn.execute(
                token_outcomes_table.insert().values(
                    outcome_id=f"out_{generate_id()[:12]}",
                    run_id="run-B",
                    token_id=token_a.token_id,
                    outcome=TerminalOutcome.SUCCESS.value,
                    path=TerminalPath.DEFAULT_FLOW.value,
                    completed=1,
                    recorded_at=now(),
                    sink_name="output",
                )
            )

    def test_schema_composite_fk_prevents_cross_run_node_state(self):
        """Schema composite FK on node_states must reject mismatched (token_id, run_id).

        The composite ForeignKeyConstraint on node_states ensures that a node_state
        row cannot reference a token from a different run, preventing cross-run
        contamination in the audit trail.
        """
        from sqlalchemy.exc import IntegrityError

        from elspeth.core.landscape._helpers import generate_id, now
        from elspeth.core.landscape.schema import node_states_table

        db = make_landscape_db()
        factory = make_factory(db)

        # Set up run-A with row + token
        factory.run_lifecycle.begin_run(config={}, canonical_version="v1", run_id="run-A")
        factory.data_flow.register_node(
            run_id="run-A",
            plugin_name="csv",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            node_id="source-0",
            schema_config=_DYNAMIC_SCHEMA,
        )
        row_a = factory.data_flow.create_row(
            run_id="run-A",
            source_node_id="source-0",
            row_index=0,
            data={"col": "value"},
            source_row_index=0,
            ingest_sequence=0,
        )
        token_a = factory.data_flow.create_token(row_a.row_id)

        # Set up run-B with its own node (but no tokens)
        factory.run_lifecycle.begin_run(config={}, canonical_version="v1", run_id="run-B")
        factory.data_flow.register_node(
            run_id="run-B",
            plugin_name="csv",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            node_id="source-0",
            schema_config=_DYNAMIC_SCHEMA,
        )

        # Try to insert a node_state with token_id from run-A but run_id from run-B
        # The composite FK should reject this
        with pytest.raises(IntegrityError), db.connection() as conn:
            conn.execute(
                node_states_table.insert().values(
                    state_id=f"state_{generate_id()[:12]}",
                    run_id="run-B",
                    token_id=token_a.token_id,
                    node_id="source-0",
                    step_index=0,
                    attempt=1,
                    status="open",
                    input_hash="fake-hash",
                    started_at=now(),
                )
            )
