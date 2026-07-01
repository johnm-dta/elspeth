"""Direct unit tests for ExecutionRepository.

Tests exercise the repository directly (not through RecorderFactory delegation)
to verify audit integrity checks, edge cases, and crash paths that the delegation
tests don't cover.

The _make_repo() helper returns (LandscapeDB, ExecutionRepository, RecorderFactory)
— the factory is used for graph setup only (register_node, create_row, create_token),
while the repo is tested directly.
"""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any, ClassVar
from unittest.mock import MagicMock

import pytest
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError

from elspeth.contracts import (
    BatchStatus,
    CallStatus,
    CallType,
    FrameworkBugError,
    NodeStateCompleted,
    NodeStateFailed,
    NodeStateOpen,
    NodeStatePending,
    NodeStateStatus,
    NodeType,
    RoutingMode,
    RoutingSpec,
    TriggerType,
)
from elspeth.contracts.call_data import RawCallPayload
from elspeth.contracts.errors import AuditIntegrityError, ConfigGateReason, ExecutionError, TransformSuccessReason
from elspeth.contracts.schema import SchemaConfig
from elspeth.core.canonical import stable_hash
from elspeth.core.landscape import LandscapeDB
from elspeth.core.landscape._database_ops import DatabaseOps
from elspeth.core.landscape.errors import LandscapeRecordError
from elspeth.core.landscape.execution_repository import ExecutionRepository
from elspeth.core.landscape.factory import RecorderFactory
from elspeth.core.landscape.model_loaders import (
    ArtifactLoader,
    BatchLoader,
    BatchMemberLoader,
    CallLoader,
    NodeStateLoader,
    OperationLoader,
    RoutingEventLoader,
)
from elspeth.core.landscape.schema import node_states_table
from elspeth.core.payload_store import FilesystemPayloadStore
from tests.fixtures.landscape import make_factory, make_landscape_db
from tests.fixtures.stores import MockPayloadStore

_DYNAMIC_SCHEMA = SchemaConfig.from_dict({"mode": "observed"})


def _make_repo(
    *,
    run_id: str = "run-1",
    payload_store: FilesystemPayloadStore | None = None,
) -> tuple[LandscapeDB, ExecutionRepository, RecorderFactory]:
    """Create an ExecutionRepository with supporting infrastructure.

    Returns (db, repo, factory) — factory is for graph setup only.
    """
    db = make_landscape_db()
    ops = DatabaseOps(db)
    repo = ExecutionRepository(
        db,
        ops,
        node_state_loader=NodeStateLoader(),
        routing_event_loader=RoutingEventLoader(),
        call_loader=CallLoader(),
        operation_loader=OperationLoader(),
        batch_loader=BatchLoader(),
        batch_member_loader=BatchMemberLoader(),
        artifact_loader=ArtifactLoader(),
        payload_store=payload_store,
    )
    factory = make_factory(db)
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
        plugin_name="transform",
        node_type=NodeType.TRANSFORM,
        plugin_version="1.0",
        config={},
        node_id="transform-1",
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
    return db, repo, factory


def _make_repo_with_token(
    *,
    run_id: str = "run-1",
    payload_store: FilesystemPayloadStore | None = None,
) -> tuple[LandscapeDB, ExecutionRepository, RecorderFactory, str]:
    """Create repo with a token ready for processing."""
    db, repo, factory = _make_repo(run_id=run_id, payload_store=payload_store)
    factory.data_flow.create_row(run_id, "source-0", 0, {"name": "test"}, row_id="row-1", source_row_index=0, ingest_sequence=0)
    factory.data_flow.create_token("row-1", token_id="tok-1")
    return db, repo, factory, "tok-1"


class _TrackingPayloadStore(MockPayloadStore):
    """Mock payload store that records store() side effects."""

    def __init__(self) -> None:
        super().__init__()
        self.store_calls: list[str] = []

    def store(self, content: bytes) -> str:
        content_hash = super().store(content)
        self.store_calls.append(content_hash)
        return content_hash


# ---------------------------------------------------------------------------
# C1: begin_node_state quarantined=True repr_hash fallback
# ---------------------------------------------------------------------------


class TestBeginNodeStateQuarantined:
    """Tests for quarantined input_data handling in begin_node_state (C1)."""

    def test_nan_input_uses_repr_hash_when_quarantined(self) -> None:
        """NaN in input_data triggers repr_hash fallback when quarantined=True.

        stable_hash rejects NaN (non-canonical per RFC 8785). The quarantine
        path catches the ValueError and falls back to repr_hash.
        """
        _db, repo, _fac, tok = _make_repo_with_token()
        data_with_nan = {"value": float("nan")}
        state = repo.begin_node_state(
            tok,
            "transform-1",
            "run-1",
            1,
            data_with_nan,
            quarantined=True,
        )
        assert isinstance(state, NodeStateOpen)
        assert state.input_hash is not None
        assert len(state.input_hash) > 0

    def test_canonical_input_uses_stable_hash_when_quarantined(self) -> None:
        """Normal input_data uses stable_hash even when quarantined=True.

        The repr_hash fallback only triggers when stable_hash raises.
        """
        _db, repo, _fac, tok = _make_repo_with_token()
        normal_data = {"value": 42}
        state = repo.begin_node_state(
            tok,
            "transform-1",
            "run-1",
            1,
            normal_data,
            quarantined=True,
        )
        assert isinstance(state, NodeStateOpen)
        # Verify it used stable_hash (canonical) not repr_hash
        from elspeth.core.canonical import stable_hash

        expected_hash = stable_hash(normal_data)
        assert state.input_hash == expected_hash

    def test_nan_input_crashes_when_not_quarantined(self) -> None:
        """NaN in input_data raises ValueError when quarantined=False.

        Without the quarantine flag, NaN is a bug in our code (Tier 2 data
        should not contain non-canonical values).
        """
        _db, repo, _fac, tok = _make_repo_with_token()
        data_with_nan = {"value": float("nan")}
        with pytest.raises(ValueError, match=r"[Nn]a[Nn]"):
            repo.begin_node_state(
                tok,
                "transform-1",
                "run-1",
                1,
                data_with_nan,
                quarantined=False,
            )


# ---------------------------------------------------------------------------
# C2: complete_node_state crash paths (post-update validation)
# ---------------------------------------------------------------------------


class TestCompleteNodeStateCrashPaths:
    """Tests for complete_node_state audit integrity checks (C2)."""

    def test_record_completed_node_state_inserts_terminal_state_atomically(self) -> None:
        """Immediate source-style completion should not require an OPEN row first."""
        db, repo, _fac, tok = _make_repo_with_token()

        state = repo.record_completed_node_state(
            token_id=tok,
            node_id="source-0",
            run_id="run-1",
            step_index=0,
            input_data={"name": "test"},
            output_data={"name": "test"},
            duration_ms=0,
        )

        assert isinstance(state, NodeStateCompleted)
        assert state.token_id == tok
        assert state.node_id == "source-0"
        assert state.step_index == 0
        assert state.status is NodeStateStatus.COMPLETED
        assert state.input_hash == stable_hash({"name": "test"})
        assert state.output_hash == stable_hash({"name": "test"})
        assert state.duration_ms == 0

        with db.read_only_connection() as conn:
            rows = conn.execute(select(node_states_table).where(node_states_table.c.state_id == state.state_id)).mappings().all()

        assert len(rows) == 1
        assert rows[0]["status"] == NodeStateStatus.COMPLETED.value
        assert rows[0]["completed_at"] is not None

    def test_batch_begin_and_complete_success_states_preserves_per_token_rows(self) -> None:
        """Batch sink-state writes must still leave one auditable row per token."""
        db, repo, fac, tok = _make_repo_with_token()
        fac.data_flow.create_row("run-1", "source-0", 1, {"name": "second"}, row_id="row-2", source_row_index=1, ingest_sequence=1)
        fac.data_flow.create_token("row-2", token_id="tok-2")

        states = repo.begin_node_states_many(
            (
                (tok, "sink-0", "run-1", 2, {"name": "test"}),
                ("tok-2", "sink-0", "run-1", 2, {"name": "second"}),
            )
        )

        assert [state.token_id for state in states] == [tok, "tok-2"]
        assert all(isinstance(state, NodeStateOpen) for state in states)
        repo.complete_node_states_completed_many(
            (
                (states[0].state_id, {"row": {"name": "test"}, "artifact_path": "out.csv", "content_hash": "hash"}, 2.5),
                (states[1].state_id, {"row": {"name": "second"}, "artifact_path": "out.csv", "content_hash": "hash"}, 2.5),
            )
        )

        with db.read_only_connection() as conn:
            rows = (
                conn.execute(select(node_states_table).where(node_states_table.c.state_id.in_([state.state_id for state in states])))
                .mappings()
                .all()
            )

        assert {row["token_id"] for row in rows} == {tok, "tok-2"}
        assert {row["status"] for row in rows} == {NodeStateStatus.COMPLETED.value}
        assert {row["output_hash"] for row in rows} == {
            stable_hash({"row": {"name": "test"}, "artifact_path": "out.csv", "content_hash": "hash"}),
            stable_hash({"row": {"name": "second"}, "artifact_path": "out.csv", "content_hash": "hash"}),
        }

    def test_batch_begin_rowcount_mismatch_rolls_back_inserted_open_states(self) -> None:
        """Rowcount mismatches must abort inside the write transaction."""
        db, repo, fac, tok = _make_repo_with_token()
        fac.data_flow.create_row("run-1", "source-0", 1, {"name": "second"}, row_id="row-2", source_row_index=1, ingest_sequence=1)
        fac.data_flow.create_token("row-2", token_id="tok-2")
        original_connection = repo._db.write_connection

        from contextlib import contextmanager

        @contextmanager
        def rowcount_mismatch_connection():
            with original_connection() as conn:
                original_execute = conn.execute

                def patched_execute(stmt, *args: Any, **kwargs: Any):
                    result = original_execute(stmt, *args, **kwargs)
                    if stmt.is_insert and stmt.table is node_states_table:
                        mock_result = MagicMock()
                        mock_result.rowcount = 1
                        return mock_result
                    return result

                conn.execute = patched_execute
                yield conn

        repo._db.write_connection = rowcount_mismatch_connection  # type: ignore[method-assign]

        with pytest.raises(LandscapeRecordError, match="affected 1 rows for 2 states"):
            repo.begin_node_states_many(
                (
                    (tok, "sink-0", "run-1", 2, {"name": "test"}),
                    ("tok-2", "sink-0", "run-1", 2, {"name": "second"}),
                )
            )

        with db.read_only_connection() as conn:
            rows = conn.execute(select(node_states_table.c.token_id).where(node_states_table.c.token_id.in_((tok, "tok-2")))).all()
        assert rows == []

    def test_batch_complete_rejects_already_terminal_state(self) -> None:
        """Batch completion must preserve terminal-state immutability checks."""
        _db, repo, _fac, tok = _make_repo_with_token()
        state = repo.begin_node_state(tok, "sink-0", "run-1", 2, {"a": 1})
        repo.complete_node_state(state.state_id, NodeStateStatus.COMPLETED, output_data={"b": 2}, duration_ms=1.0)

        with pytest.raises(AuditIntegrityError, match="already terminal"):
            repo.complete_node_states_completed_many(((state.state_id, {"b": 3}, 2.0),))

    def test_nonexistent_state_raises_audit_integrity(self) -> None:
        """Completing a nonexistent state must raise AuditIntegrityError.

        The rowcount check in the single-transaction UPDATE catches this.
        """
        _db, repo, _fac, _tok = _make_repo_with_token()
        with pytest.raises(AuditIntegrityError, match="zero rows affected"):
            repo.complete_node_state(
                "nonexistent-state",
                NodeStateStatus.COMPLETED,
                output_data={"result": "ok"},
                duration_ms=10.0,
            )

    def test_completed_node_state_rewrite_raises(self) -> None:
        """Cannot overwrite a COMPLETED node state.

        Regression test for elspeth-2c99c9a451: complete_node_state() lacked the
        terminal-state guard, allowing a completed node state to be silently
        rewritten to FAILED (audit immutability violation).
        """
        _db, repo, _fac, tok = _make_repo_with_token()
        repo.begin_node_state(tok, "transform-1", "run-1", 1, {"a": 1}, state_id="state-term-c", attempt=0)
        repo.complete_node_state("state-term-c", NodeStateStatus.COMPLETED, output_data={"b": 2}, duration_ms=5.0)

        from elspeth.contracts.errors import ExecutionError

        with pytest.raises(AuditIntegrityError, match="already terminal"):
            repo.complete_node_state(
                "state-term-c",
                NodeStateStatus.FAILED,
                error=ExecutionError(exception="late failure", exception_type="RuntimeError"),
                duration_ms=1.0,
            )

    def test_failed_node_state_rewrite_raises(self) -> None:
        """Cannot overwrite a FAILED node state.

        Same invariant as test_completed_node_state_rewrite_raises but in the
        FAILED→COMPLETED direction.
        """
        from elspeth.contracts.errors import ExecutionError

        _db, repo, _fac, tok = _make_repo_with_token()
        repo.begin_node_state(tok, "transform-1", "run-1", 1, {"a": 1}, state_id="state-term-f", attempt=0)
        error = ExecutionError(exception="first failure", exception_type="ValueError")
        repo.complete_node_state("state-term-f", NodeStateStatus.FAILED, error=error, duration_ms=1.0)

        with pytest.raises(AuditIntegrityError, match="already terminal"):
            repo.complete_node_state("state-term-f", NodeStateStatus.COMPLETED, output_data={"b": 2}, duration_ms=5.0)

    def test_pending_node_state_can_be_completed(self) -> None:
        """PENDING is non-terminal — completing a PENDING state to COMPLETED must succeed."""
        _db, repo, _fac, tok = _make_repo_with_token()
        repo.begin_node_state(tok, "transform-1", "run-1", 1, {"a": 1}, state_id="state-pend", attempt=0)
        repo.complete_node_state("state-pend", NodeStateStatus.PENDING, duration_ms=3.0)
        result = repo.complete_node_state("state-pend", NodeStateStatus.COMPLETED, output_data={"b": 2}, duration_ms=5.0)
        assert isinstance(result, NodeStateCompleted)

    def test_complete_returns_typed_union(self) -> None:
        """complete_node_state returns the correct typed variant for each status."""
        _db, repo, _fac, tok = _make_repo_with_token()

        # COMPLETED (attempt=0)
        repo.begin_node_state(tok, "transform-1", "run-1", 1, {"a": 1}, state_id="state-c", attempt=0)
        result_c = repo.complete_node_state("state-c", NodeStateStatus.COMPLETED, output_data={"b": 2}, duration_ms=5.0)
        assert isinstance(result_c, NodeStateCompleted)

        # PENDING (attempt=1 to avoid UNIQUE constraint)
        repo.begin_node_state(tok, "transform-1", "run-1", 2, {"a": 1}, state_id="state-p", attempt=1)
        result_p = repo.complete_node_state("state-p", NodeStateStatus.PENDING, duration_ms=3.0)
        assert isinstance(result_p, NodeStatePending)

        # FAILED (attempt=2 to avoid UNIQUE constraint)
        from elspeth.contracts.errors import ExecutionError

        repo.begin_node_state(tok, "transform-1", "run-1", 3, {"a": 1}, state_id="state-f", attempt=2)
        error = ExecutionError(exception="test failure", exception_type="ValueError")
        result_f = repo.complete_node_state("state-f", NodeStateStatus.FAILED, error=error, duration_ms=1.0)
        assert isinstance(result_f, NodeStateFailed)


class TestCompletedRowLookup:
    """Exact completed-row lookup for coalesce late-arrival detection."""

    def test_has_completed_row_for_node_scopes_by_run_node_and_row(self) -> None:
        _db, repo, fac, tok = _make_repo_with_token()
        fac.data_flow.create_row("run-1", "source-0", 1, {"name": "second"}, row_id="row-2", source_row_index=1, ingest_sequence=1)
        fac.data_flow.create_token("row-2", token_id="tok-2")

        first = repo.begin_node_state(tok, "transform-1", "run-1", 1, {"name": "test"})
        second = repo.begin_node_state("tok-2", "transform-1", "run-1", 1, {"name": "second"})
        repo.complete_node_state(first.state_id, NodeStateStatus.COMPLETED, output_data={"ok": True}, duration_ms=1.0)
        repo.complete_node_state(second.state_id, NodeStateStatus.COMPLETED, output_data={"ok": True}, duration_ms=1.0)

        assert repo.has_completed_row_for_node(run_id="run-1", node_id="transform-1", row_id="row-1") is True
        assert repo.has_completed_row_for_node(run_id="run-1", node_id="transform-1", row_id="row-2") is True
        assert repo.has_completed_row_for_node(run_id="run-1", node_id="transform-1", row_id="row-missing") is False
        assert repo.has_completed_row_for_node(run_id="run-1", node_id="sink-0", row_id="row-1") is False
        assert repo.has_completed_row_for_node(run_id="run-missing", node_id="transform-1", row_id="row-1") is False

    def test_has_completed_row_for_node_ignores_open_state(self) -> None:
        _db, repo, _fac, tok = _make_repo_with_token()
        repo.begin_node_state(tok, "transform-1", "run-1", 1, {"name": "test"})

        assert repo.has_completed_row_for_node(run_id="run-1", node_id="transform-1", row_id="row-1") is False


class TestCompleteNodeStateForbiddenFields:
    """Regression tests for elspeth-22e2bca0c1: forbidden fields per status."""

    def test_pending_rejects_output_data(self) -> None:
        _db, repo, _fac, tok = _make_repo_with_token()
        repo.begin_node_state(tok, "transform-1", "run-1", 1, {"a": 1}, state_id="s-1")
        with pytest.raises(ValueError, match=r"PENDING.*must not have output_data"):
            repo.complete_node_state("s-1", NodeStateStatus.PENDING, output_data={"x": 1}, duration_ms=5.0)

    def test_pending_rejects_error(self) -> None:
        from elspeth.contracts.errors import ExecutionError

        _db, repo, _fac, tok = _make_repo_with_token()
        repo.begin_node_state(tok, "transform-1", "run-1", 1, {"a": 1}, state_id="s-1")
        err = ExecutionError(exception="oops", exception_type="ValueError")
        with pytest.raises(ValueError, match=r"PENDING.*must not have error"):
            repo.complete_node_state("s-1", NodeStateStatus.PENDING, error=err, duration_ms=5.0)

    def test_pending_rejects_success_reason(self) -> None:
        _db, repo, _fac, tok = _make_repo_with_token()
        repo.begin_node_state(tok, "transform-1", "run-1", 1, {"a": 1}, state_id="s-1")
        with pytest.raises(ValueError, match=r"PENDING.*must not have success_reason"):
            repo.complete_node_state("s-1", NodeStateStatus.PENDING, success_reason={"reason": "ok"}, duration_ms=5.0)  # type: ignore[call-overload]  # intentionally invalid: testing rejection of success_reason with PENDING

    def test_completed_rejects_error(self) -> None:
        from elspeth.contracts.errors import ExecutionError

        _db, repo, _fac, tok = _make_repo_with_token()
        repo.begin_node_state(tok, "transform-1", "run-1", 1, {"a": 1}, state_id="s-1")
        err = ExecutionError(exception="oops", exception_type="ValueError")
        with pytest.raises(ValueError, match=r"COMPLETED.*must not have error"):
            repo.complete_node_state("s-1", NodeStateStatus.COMPLETED, output_data={"x": 1}, error=err, duration_ms=5.0)

    def test_failed_rejects_success_reason(self) -> None:
        from elspeth.contracts.errors import ExecutionError

        _db, repo, _fac, tok = _make_repo_with_token()
        repo.begin_node_state(tok, "transform-1", "run-1", 1, {"a": 1}, state_id="s-1")
        err = ExecutionError(exception="oops", exception_type="ValueError")
        with pytest.raises(ValueError, match=r"FAILED.*must not have success_reason"):
            repo.complete_node_state("s-1", NodeStateStatus.FAILED, error=err, success_reason={"reason": "ok"}, duration_ms=5.0)  # type: ignore[call-overload]  # intentionally invalid: testing rejection of success_reason with FAILED


# ---------------------------------------------------------------------------
# H5: record_routing_events rowcount=0 AuditIntegrityError
# ---------------------------------------------------------------------------


class TestRecordRoutingEventsRowcount:
    """Test that record_routing_events checks INSERT rowcount (H5)."""

    def test_zero_rowcount_raises_audit_integrity(self) -> None:
        """Mocked zero rowcount on routing event INSERT raises AuditIntegrityError.

        This simulates a database anomaly where the INSERT succeeds but
        reports zero rows affected.
        """
        _db, repo, fac, tok = _make_repo_with_token()

        # Create a node state and edge so the routing event has valid references
        state = repo.begin_node_state(tok, "transform-1", "run-1", 1, {"a": 1})
        fac.data_flow.register_edge(
            run_id="run-1",
            from_node_id="transform-1",
            to_node_id="sink-0",
            label="continue",
            mode=RoutingMode.MOVE,
            edge_id="edge-1",
        )

        routes = [RoutingSpec(edge_id="edge-1", mode=RoutingMode.MOVE)]

        # Mock the connection's execute to return rowcount=0 for INSERTs
        original_connection = repo._db.write_connection

        from contextlib import contextmanager

        @contextmanager
        def mock_connection():
            with original_connection() as conn:
                original_execute = conn.execute

                def patched_execute(stmt, *args: Any, **kwargs: Any):
                    result = original_execute(stmt, *args, **kwargs)
                    # Intercept INSERT results to simulate zero rowcount
                    if stmt.is_insert:
                        mock_result = MagicMock()
                        mock_result.rowcount = 0
                        return mock_result
                    return result

                conn.execute = patched_execute
                yield conn

        repo._db.write_connection = mock_connection  # type: ignore[method-assign]

        with pytest.raises(AuditIntegrityError, match="zero rows affected"):
            repo.record_routing_events(state.state_id, routes)


# ---------------------------------------------------------------------------
# Happy path: begin + complete node state roundtrip
# ---------------------------------------------------------------------------


class TestBeginAndCompleteNodeState:
    """Happy path roundtrip for node state recording via repo."""

    def test_begin_and_complete_roundtrip(self) -> None:
        """Begin a node state, complete it, verify all fields."""
        _db, repo, _fac, tok = _make_repo_with_token()
        state = repo.begin_node_state(tok, "transform-1", "run-1", 1, {"x": 1})
        assert isinstance(state, NodeStateOpen)
        assert state.status == NodeStateStatus.OPEN
        assert state.token_id == tok

        completed = repo.complete_node_state(
            state.state_id,
            NodeStateStatus.COMPLETED,
            output_data={"y": 2},
            duration_ms=42.0,
        )
        assert isinstance(completed, NodeStateCompleted)
        assert completed.status == NodeStateStatus.COMPLETED
        assert completed.duration_ms == 42.0
        assert completed.output_hash is not None


# ---------------------------------------------------------------------------
# Call recording through repo
# ---------------------------------------------------------------------------


class TestRecordCall:
    """Basic call recording through ExecutionRepository."""

    def test_record_call_roundtrip(self) -> None:
        """Record a call and verify it's retrievable."""
        _db, repo, _fac, tok = _make_repo_with_token()
        state = repo.begin_node_state(tok, "transform-1", "run-1", 1, {"a": 1})

        idx = repo.allocate_call_index(state.state_id)
        assert idx == 0

        call = repo.record_call(
            state.state_id,
            idx,
            CallType.LLM,
            CallStatus.SUCCESS,
            RawCallPayload({"prompt": "hello"}),
            RawCallPayload({"response": "world"}),
        )
        assert call.call_type == CallType.LLM
        assert call.status == CallStatus.SUCCESS


# ---------------------------------------------------------------------------
# Batch lifecycle through repo
# ---------------------------------------------------------------------------


class TestBatchLifecycle:
    """Batch create → add member → complete lifecycle."""

    def test_batch_lifecycle(self) -> None:
        """Create batch, add members, complete it."""
        _db, repo, _fac, tok = _make_repo_with_token()

        batch = repo.create_batch("run-1", "agg-1")
        assert batch.status == BatchStatus.DRAFT

        member = repo.add_batch_member(batch.batch_id, tok, 0)
        assert member.ordinal == 0

        completed = repo.complete_batch(
            batch.batch_id,
            BatchStatus.COMPLETED,
            trigger_type=TriggerType.COUNT,
            trigger_reason="count=1",
        )
        assert completed.status == BatchStatus.COMPLETED
        assert completed.trigger_type == TriggerType.COUNT


# ---------------------------------------------------------------------------
# H1: complete_operation output_data_ref rowcount check
# ---------------------------------------------------------------------------


class TestCompleteOperationRowcount:
    """Test that complete_operation checks rowcount on the output_data_ref UPDATE (H1)."""

    def test_complete_operation_basic(self) -> None:
        """complete_operation without payload_store skips the second UPDATE."""
        _db, repo, _fac, _tok = _make_repo_with_token()
        op = repo.begin_operation("run-1", "source-0", "source_load")
        repo.complete_operation(op.operation_id, "completed", duration_ms=10.0)
        result = repo.get_operation(op.operation_id)
        assert result is not None
        assert result.status == "completed"


# ---------------------------------------------------------------------------
# complete_operation with payload store
# ---------------------------------------------------------------------------


class TestCompleteOperationWithPayloadStore:
    """Tests for complete_operation with and without payload store."""

    def test_complete_operation_with_output_data_no_payload_store(self) -> None:
        """complete_operation with output_data but no payload_store stores hash only."""
        _db, repo, _fac, _tok = _make_repo_with_token()
        op = repo.begin_operation("run-1", "source-0", "source_load")
        repo.complete_operation(
            op.operation_id,
            "completed",
            output_data={"rows_loaded": 100},
            duration_ms=50.0,
        )
        result = repo.get_operation(op.operation_id)
        assert result is not None
        assert result.status == "completed"
        assert result.output_data_hash is not None
        # No payload store means no output_data_ref
        assert result.output_data_ref is None

    def test_complete_operation_with_payload_store(self, tmp_path: Path) -> None:
        """complete_operation with payload_store persists output_data to store."""
        store = FilesystemPayloadStore(tmp_path / "payloads")
        _db, repo, _fac, _tok = _make_repo_with_token(payload_store=store)
        op = repo.begin_operation("run-1", "source-0", "source_load")
        repo.complete_operation(
            op.operation_id,
            "completed",
            output_data={"rows_loaded": 42},
            duration_ms=15.0,
        )
        result = repo.get_operation(op.operation_id)
        assert result is not None
        assert result.status == "completed"
        assert result.output_data_hash is not None
        assert result.output_data_ref is not None
        # Verify the payload was actually stored
        payload_bytes = store.retrieve(result.output_data_ref)
        assert b"rows_loaded" in payload_bytes

    def test_complete_operation_failed_status(self) -> None:
        """complete_operation with failed status records error message."""
        _db, repo, _fac, _tok = _make_repo_with_token()
        op = repo.begin_operation("run-1", "source-0", "source_load")
        repo.complete_operation(
            op.operation_id,
            "failed",
            error="Connection refused",
            duration_ms=100.0,
        )
        result = repo.get_operation(op.operation_id)
        assert result is not None
        assert result.status == "failed"
        assert result.error_message == "Connection refused"

    def test_complete_operation_failed_status_rejects_empty_error_message(self) -> None:
        """complete_operation must reject blank error strings for failed operations."""
        _db, repo, _fac, _tok = _make_repo_with_token()
        op = repo.begin_operation("run-1", "source-0", "source_load")

        with pytest.raises(FrameworkBugError, match="error must not be empty"):
            repo.complete_operation(
                op.operation_id,
                "failed",
                error="",
                duration_ms=100.0,
            )

    @pytest.mark.parametrize("invalid_status", ["open", "bogus", "pending"])
    def test_complete_operation_rejects_invalid_runtime_statuses(self, invalid_status: str) -> None:
        """complete_operation must reject runtime status strings outside the terminal allowlist."""
        _db, repo, _fac, _tok = _make_repo_with_token()
        op = repo.begin_operation("run-1", "source-0", "source_load")

        with pytest.raises(FrameworkBugError, match="unsupported status"):
            repo.complete_operation(
                op.operation_id,
                invalid_status,  # type: ignore[arg-type]  # Intentional runtime-boundary test
                duration_ms=5.0,
            )

    def test_complete_nonexistent_operation_raises_framework_bug(self) -> None:
        """Completing a nonexistent operation raises FrameworkBugError."""
        _db, repo, _fac, _tok = _make_repo_with_token()
        with pytest.raises(FrameworkBugError, match="non-existent"):
            repo.complete_operation("op_doesnotexist", "completed", duration_ms=10.0)

    def test_complete_already_completed_operation_raises_framework_bug(self) -> None:
        """Completing an already-completed operation raises FrameworkBugError."""
        _db, repo, _fac, _tok = _make_repo_with_token()
        op = repo.begin_operation("run-1", "source-0", "source_load")
        repo.complete_operation(op.operation_id, "completed", duration_ms=10.0)
        with pytest.raises(FrameworkBugError, match="already-completed"):
            repo.complete_operation(op.operation_id, "completed", duration_ms=10.0)

    def test_begin_operation_with_input_data_and_payload_store(self, tmp_path: Path) -> None:
        """begin_operation with input_data and payload_store persists input."""
        store = FilesystemPayloadStore(tmp_path / "payloads")
        _db, repo, _fac, _tok = _make_repo_with_token(payload_store=store)
        op = repo.begin_operation(
            "run-1",
            "source-0",
            "source_load",
            input_data={"source_path": "/data/input.csv"},
        )
        assert op.input_data_hash is not None
        assert op.input_data_ref is not None

    def test_begin_operation_with_input_data_no_payload_store(self) -> None:
        """begin_operation with input_data but no payload_store stores hash only."""
        _db, repo, _fac, _tok = _make_repo_with_token()
        op = repo.begin_operation(
            "run-1",
            "source-0",
            "source_load",
            input_data={"source_path": "/data/input.csv"},
        )
        assert op.input_data_hash is not None
        assert op.input_data_ref is None


# ---------------------------------------------------------------------------
# find_call_by_request_hash
# ---------------------------------------------------------------------------


class TestFindCallByRequestHash:
    """Tests for find_call_by_request_hash (replay mode lookup)."""

    def test_find_call_found(self) -> None:
        """Find a previously recorded call by its request hash."""
        _db, repo, _fac, tok = _make_repo_with_token()
        state = repo.begin_node_state(tok, "transform-1", "run-1", 1, {"a": 1})
        request_data = RawCallPayload({"prompt": "classify this"})
        idx = repo.allocate_call_index(state.state_id)
        recorded_call = repo.record_call(
            state.state_id,
            idx,
            CallType.LLM,
            CallStatus.SUCCESS,
            request_data,
            RawCallPayload({"result": "positive"}),
        )

        found = repo.find_call_by_request_hash(
            "run-1",
            CallType.LLM,
            recorded_call.request_hash,
        )
        assert found is not None
        assert found.call_id == recorded_call.call_id
        assert found.request_hash == recorded_call.request_hash

    def test_find_call_not_found(self) -> None:
        """Return None when no call matches the request hash."""
        _db, repo, _fac, tok = _make_repo_with_token()
        # Create a state so the run has node_states, but no calls
        repo.begin_node_state(tok, "transform-1", "run-1", 1, {"a": 1})

        result = repo.find_call_by_request_hash(
            "run-1",
            CallType.LLM,
            "nonexistent_hash_0000000000000000000000000000000000000000000000000000",
        )
        assert result is None

    def test_find_call_wrong_type_not_found(self) -> None:
        """Return None when call type doesn't match even if hash matches."""
        _db, repo, _fac, tok = _make_repo_with_token()
        state = repo.begin_node_state(tok, "transform-1", "run-1", 1, {"a": 1})
        request_data = RawCallPayload({"url": "https://api.example.com"})
        idx = repo.allocate_call_index(state.state_id)
        recorded = repo.record_call(
            state.state_id,
            idx,
            CallType.HTTP,
            CallStatus.SUCCESS,
            request_data,
            RawCallPayload({"status": 200}),
        )

        # Search with LLM type but HTTP hash
        result = repo.find_call_by_request_hash(
            "run-1",
            CallType.LLM,
            recorded.request_hash,
        )
        assert result is None

    def test_find_call_sequence_index_for_duplicates(self) -> None:
        """sequence_index disambiguates duplicate request hashes (retries)."""
        _db, repo, _fac, tok = _make_repo_with_token()
        state = repo.begin_node_state(tok, "transform-1", "run-1", 1, {"a": 1})
        request_data = RawCallPayload({"prompt": "same request"})

        # Record same request 3 times with different responses
        call_ids = []
        for i in range(3):
            idx = repo.allocate_call_index(state.state_id)
            call = repo.record_call(
                state.state_id,
                idx,
                CallType.LLM,
                CallStatus.SUCCESS,
                request_data,
                RawCallPayload({"response": f"response-{i}"}),
            )
            call_ids.append(call.call_id)

        request_hash = stable_hash(request_data.to_dict())

        # sequence_index=0 gets the first
        first = repo.find_call_by_request_hash("run-1", CallType.LLM, request_hash, sequence_index=0)
        assert first is not None
        assert first.call_id == call_ids[0]

        # sequence_index=1 gets the second
        second = repo.find_call_by_request_hash("run-1", CallType.LLM, request_hash, sequence_index=1)
        assert second is not None
        assert second.call_id == call_ids[1]

        # sequence_index=2 gets the third
        third = repo.find_call_by_request_hash("run-1", CallType.LLM, request_hash, sequence_index=2)
        assert third is not None
        assert third.call_id == call_ids[2]

        # sequence_index=3 returns None (no 4th occurrence)
        fourth = repo.find_call_by_request_hash("run-1", CallType.LLM, request_hash, sequence_index=3)
        assert fourth is None


# ---------------------------------------------------------------------------
# retry_batch
# ---------------------------------------------------------------------------


class TestRetryBatch:
    """Tests for retry_batch mechanics."""

    def test_retry_failed_batch_creates_new_draft(self) -> None:
        """Retrying a failed batch creates a new draft batch with incremented attempt."""
        _db, repo, _fac, tok = _make_repo_with_token()

        # Create and fail a batch
        batch = repo.create_batch("run-1", "agg-1", batch_id="batch-orig")
        repo.add_batch_member(batch.batch_id, tok, 0)
        repo.complete_batch(batch.batch_id, BatchStatus.FAILED, trigger_type=TriggerType.COUNT, trigger_reason="err")

        # Retry
        retry = repo.retry_batch(batch.batch_id)
        assert retry.batch_id != batch.batch_id
        assert retry.status == BatchStatus.DRAFT
        assert retry.attempt == 1  # original was 0, retry is 1
        assert retry.aggregation_node_id == "agg-1"
        assert retry.run_id == "run-1"

    def test_retry_batch_copies_members(self) -> None:
        """retry_batch copies all members from the original batch."""
        _db, repo, fac, tok = _make_repo_with_token()
        fac.data_flow.create_token("row-1", token_id="tok-2")

        batch = repo.create_batch("run-1", "agg-1")
        repo.add_batch_member(batch.batch_id, tok, 0)
        repo.add_batch_member(batch.batch_id, "tok-2", 1)
        repo.complete_batch(batch.batch_id, BatchStatus.FAILED, trigger_type=TriggerType.COUNT, trigger_reason="err")

        retry = repo.retry_batch(batch.batch_id)
        members = repo.get_batch_members(retry.batch_id)
        assert len(members) == 2
        assert members[0].token_id == tok
        assert members[0].ordinal == 0
        assert members[1].token_id == "tok-2"
        assert members[1].ordinal == 1

    def test_retry_batch_idempotent(self) -> None:
        """Calling retry_batch twice returns the same retry batch (no duplicates)."""
        _db, repo, _fac, tok = _make_repo_with_token()

        batch = repo.create_batch("run-1", "agg-1")
        repo.add_batch_member(batch.batch_id, tok, 0)
        repo.complete_batch(batch.batch_id, BatchStatus.FAILED, trigger_type=TriggerType.COUNT, trigger_reason="err")

        retry1 = repo.retry_batch(batch.batch_id)
        retry2 = repo.retry_batch(batch.batch_id)
        assert retry1.batch_id == retry2.batch_id
        assert retry1.attempt == retry2.attempt

    def test_retry_batch_keeps_lineages_distinct_for_multiple_failed_batches(self) -> None:
        """Failed batches on the same aggregation node must not share one retry row."""
        _db, repo, fac, tok = _make_repo_with_token()
        fac.data_flow.create_row("run-1", "source-0", 1, {"name": "second"}, row_id="row-2", source_row_index=1, ingest_sequence=1)
        fac.data_flow.create_token("row-2", token_id="tok-2")

        batch_a = repo.create_batch("run-1", "agg-1", batch_id="batch-a")
        repo.add_batch_member(batch_a.batch_id, tok, 0)
        repo.complete_batch(batch_a.batch_id, BatchStatus.FAILED, trigger_type=TriggerType.COUNT, trigger_reason="fail-a")
        retry_a = repo.retry_batch(batch_a.batch_id)

        batch_b = repo.create_batch("run-1", "agg-1", batch_id="batch-b")
        repo.add_batch_member(batch_b.batch_id, "tok-2", 0)
        repo.complete_batch(batch_b.batch_id, BatchStatus.FAILED, trigger_type=TriggerType.COUNT, trigger_reason="fail-b")
        retry_b = repo.retry_batch(batch_b.batch_id)

        assert retry_b.batch_id != retry_a.batch_id
        assert [member.token_id for member in repo.get_batch_members(retry_b.batch_id)] == ["tok-2"]

    def test_retry_non_failed_batch_raises_audit_integrity_error(self) -> None:
        """Cannot retry a batch that isn't in FAILED status."""
        _db, repo, _fac, tok = _make_repo_with_token()

        batch = repo.create_batch("run-1", "agg-1")
        repo.add_batch_member(batch.batch_id, tok, 0)
        # Batch is still DRAFT, not FAILED
        with pytest.raises(AuditIntegrityError, match="can only retry failed batches"):
            repo.retry_batch(batch.batch_id)

    def test_retry_nonexistent_batch_raises_audit_integrity_error(self) -> None:
        """Retrying a nonexistent batch raises AuditIntegrityError."""
        _db, repo, _fac, _tok = _make_repo_with_token()
        with pytest.raises(AuditIntegrityError, match="not found"):
            repo.retry_batch("nonexistent-batch")


# ---------------------------------------------------------------------------
# Operation call recording
# ---------------------------------------------------------------------------


class TestRecordOperationCall:
    """Tests for record_operation_call and allocate_operation_call_index."""

    def test_record_operation_call_basic(self) -> None:
        """Record an external call attributed to an operation (not a node state)."""
        _db, repo, _fac, _tok = _make_repo_with_token()
        op = repo.begin_operation("run-1", "source-0", "source_load")

        call = repo.record_operation_call(
            op.operation_id,
            CallType.HTTP,
            CallStatus.SUCCESS,
            RawCallPayload({"url": "https://api.example.com/data"}),
            RawCallPayload({"status": 200, "body": "ok"}),
            latency_ms=150.0,
        )
        assert call.call_type == CallType.HTTP
        assert call.status == CallStatus.SUCCESS
        assert call.state_id is None  # Operation call, not state call
        assert call.operation_id == op.operation_id
        assert call.latency_ms == 150.0

    def test_operation_call_index_sequential(self) -> None:
        """Operation call indices are allocated sequentially starting at 0."""
        _db, repo, _fac, _tok = _make_repo_with_token()
        op = repo.begin_operation("run-1", "source-0", "source_load")

        idx0 = repo.allocate_operation_call_index(op.operation_id)
        idx1 = repo.allocate_operation_call_index(op.operation_id)
        idx2 = repo.allocate_operation_call_index(op.operation_id)

        assert idx0 == 0
        assert idx1 == 1
        assert idx2 == 2

    def test_operation_call_independent_indices(self) -> None:
        """Different operations have independent call index sequences."""
        _db, repo, _fac, _tok = _make_repo_with_token()
        op1 = repo.begin_operation("run-1", "source-0", "source_load")
        op2 = repo.begin_operation("run-1", "sink-0", "sink_write")

        idx_op1_0 = repo.allocate_operation_call_index(op1.operation_id)
        idx_op2_0 = repo.allocate_operation_call_index(op2.operation_id)
        idx_op1_1 = repo.allocate_operation_call_index(op1.operation_id)

        assert idx_op1_0 == 0
        assert idx_op2_0 == 0
        assert idx_op1_1 == 1

    def test_record_operation_call_with_error(self) -> None:
        """Record an operation call with error status and error payload."""
        _db, repo, _fac, _tok = _make_repo_with_token()
        op = repo.begin_operation("run-1", "source-0", "source_load")

        call = repo.record_operation_call(
            op.operation_id,
            CallType.HTTP,
            CallStatus.ERROR,
            RawCallPayload({"url": "https://api.example.com/data"}),
            error=RawCallPayload({"error": "connection_refused", "code": 503}),
            latency_ms=5000.0,
        )
        assert call.status == CallStatus.ERROR
        assert call.error_json is not None
        assert "connection_refused" in call.error_json
        # No response for error calls
        assert call.response_hash is None

    def test_get_operation_calls_returns_ordered_list(self) -> None:
        """get_operation_calls returns calls ordered by call_index."""
        _db, repo, _fac, _tok = _make_repo_with_token()
        op = repo.begin_operation("run-1", "source-0", "source_load")

        for i in range(3):
            repo.record_operation_call(
                op.operation_id,
                CallType.HTTP,
                CallStatus.SUCCESS,
                RawCallPayload({"request": i}),
                RawCallPayload({"response": i}),
            )

        calls = repo.get_operation_calls(op.operation_id)
        assert len(calls) == 3
        assert [c.call_index for c in calls] == [0, 1, 2]

    def test_record_operation_call_with_payload_store(self, tmp_path: Path) -> None:
        """Operation calls auto-persist request/response to payload store."""
        store = FilesystemPayloadStore(tmp_path / "payloads")
        _db, repo, _fac, _tok = _make_repo_with_token(payload_store=store)
        op = repo.begin_operation("run-1", "source-0", "source_load")

        call = repo.record_operation_call(
            op.operation_id,
            CallType.HTTP,
            CallStatus.SUCCESS,
            RawCallPayload({"url": "https://example.com"}),
            RawCallPayload({"body": "response data"}),
        )
        # Both refs should be set when payload store is available
        assert call.request_ref is not None
        assert call.response_ref is not None


# ---------------------------------------------------------------------------
# complete_node_state success / failure transitions
# ---------------------------------------------------------------------------


class TestCompleteNodeStateSuccessFailure:
    """Tests for complete_node_state_success and failure transitions."""

    def test_complete_with_success_reason(self) -> None:
        """complete_node_state with success_reason serializes it to JSON."""
        _db, repo, _fac, tok = _make_repo_with_token()
        state = repo.begin_node_state(tok, "transform-1", "run-1", 1, {"x": 1})

        success_reason: TransformSuccessReason = {"action": "classified", "fields_modified": ["category"]}
        completed = repo.complete_node_state(
            state.state_id,
            NodeStateStatus.COMPLETED,
            output_data={"x": 1, "category": "A"},
            duration_ms=10.0,
            success_reason=success_reason,
        )
        assert isinstance(completed, NodeStateCompleted)
        assert completed.success_reason_json is not None
        assert "classified" in completed.success_reason_json

    def test_complete_rejects_success_reason_without_action(self) -> None:
        """Tier 1 success_reason writes require an action string."""
        _db, repo, _fac, tok = _make_repo_with_token()
        state = repo.begin_node_state(tok, "transform-1", "run-1", 1, {"x": 1})

        with pytest.raises(ValueError, match=r"success_reason.*action"):
            repo.complete_node_state(
                state.state_id,
                NodeStateStatus.COMPLETED,
                output_data={"x": 1},
                duration_ms=10.0,
                success_reason={"fields_added": ["x"]},  # type: ignore[typeddict-item]
            )

    def test_complete_rejects_success_reason_with_non_string_action(self) -> None:
        """Tier 1 success_reason action must be a string."""
        _db, repo, _fac, tok = _make_repo_with_token()
        state = repo.begin_node_state(tok, "transform-1", "run-1", 1, {"x": 1})

        with pytest.raises(ValueError, match=r"success_reason.*action.*str"):
            repo.complete_node_state(
                state.state_id,
                NodeStateStatus.COMPLETED,
                output_data={"x": 1},
                duration_ms=10.0,
                success_reason={"action": 123},  # type: ignore[typeddict-item]
            )

    def test_complete_failed_requires_error(self) -> None:
        """FAILED status without error raises ValueError."""
        _db, repo, _fac, tok = _make_repo_with_token()
        state = repo.begin_node_state(tok, "transform-1", "run-1", 1, {"x": 1})
        with pytest.raises(ValueError, match="FAILED node state requires error details"):
            repo.complete_node_state(
                state.state_id,
                NodeStateStatus.FAILED,
                duration_ms=5.0,
            )

    def test_complete_completed_requires_output_data(self) -> None:
        """COMPLETED status without output_data raises ValueError."""
        _db, repo, _fac, tok = _make_repo_with_token()
        state = repo.begin_node_state(tok, "transform-1", "run-1", 1, {"x": 1})
        with pytest.raises(ValueError, match="COMPLETED node state requires output_data"):
            repo.complete_node_state(
                state.state_id,
                NodeStateStatus.COMPLETED,
                duration_ms=5.0,
            )

    def test_complete_with_open_status_raises(self) -> None:
        """Cannot complete a node state with OPEN status."""
        _db, repo, _fac, tok = _make_repo_with_token()
        state = repo.begin_node_state(tok, "transform-1", "run-1", 1, {"x": 1})
        with pytest.raises(ValueError, match="Cannot complete a node state with status OPEN"):
            repo.complete_node_state(
                state.state_id,
                NodeStateStatus.OPEN,  # type: ignore[call-overload]  # Intentionally testing invalid status
                duration_ms=5.0,
            )

    def test_complete_requires_duration_ms(self) -> None:
        """duration_ms is required when completing a node state."""
        _db, repo, _fac, tok = _make_repo_with_token()
        state = repo.begin_node_state(tok, "transform-1", "run-1", 1, {"x": 1})
        with pytest.raises(ValueError, match="duration_ms is required"):
            repo.complete_node_state(
                state.state_id,
                NodeStateStatus.COMPLETED,
                output_data={"y": 2},
            )

    def test_complete_failed_with_execution_error(self) -> None:
        """complete_node_state with FAILED status and ExecutionError records error JSON."""
        _db, repo, _fac, tok = _make_repo_with_token()
        state = repo.begin_node_state(tok, "transform-1", "run-1", 1, {"x": 1})
        error = ExecutionError(exception="division by zero", exception_type="ZeroDivisionError")
        result = repo.complete_node_state(
            state.state_id,
            NodeStateStatus.FAILED,
            error=error,
            duration_ms=2.0,
        )
        assert isinstance(result, NodeStateFailed)
        assert result.error_json is not None
        assert "division by zero" in result.error_json

    def test_get_node_state_returns_none_for_missing(self) -> None:
        """get_node_state returns None for a nonexistent state_id."""
        _db, repo, _fac, _tok = _make_repo_with_token()
        result = repo.get_node_state("nonexistent-state")
        assert result is None

    def test_get_node_state_returns_open_state(self) -> None:
        """get_node_state returns the correct NodeStateOpen for an open state."""
        _db, repo, _fac, tok = _make_repo_with_token()
        state = repo.begin_node_state(tok, "transform-1", "run-1", 1, {"x": 1})
        fetched = repo.get_node_state(state.state_id)
        assert fetched is not None
        assert isinstance(fetched, NodeStateOpen)
        assert fetched.state_id == state.state_id


# ---------------------------------------------------------------------------
# record_routing_event
# ---------------------------------------------------------------------------


class TestRecordRoutingEvent:
    """Tests for single routing event recording."""

    def test_record_routing_event_basic(self) -> None:
        """Record a single routing event and verify all fields."""
        _db, repo, fac, tok = _make_repo_with_token()
        state = repo.begin_node_state(tok, "transform-1", "run-1", 1, {"a": 1})
        fac.data_flow.register_edge(
            run_id="run-1",
            from_node_id="transform-1",
            to_node_id="sink-0",
            label="continue",
            mode=RoutingMode.MOVE,
            edge_id="edge-1",
        )

        event = repo.record_routing_event(
            state.state_id,
            "edge-1",
            RoutingMode.MOVE,
        )
        assert event.state_id == state.state_id
        assert event.edge_id == "edge-1"
        assert event.mode == RoutingMode.MOVE
        assert event.ordinal == 0
        assert event.event_id is not None
        assert event.routing_group_id is not None

    def test_record_routing_event_with_reason(self) -> None:
        """Routing event with reason stores reason_hash."""
        _db, repo, fac, tok = _make_repo_with_token()
        state = repo.begin_node_state(tok, "transform-1", "run-1", 1, {"a": 1})
        fac.data_flow.register_edge(
            run_id="run-1",
            from_node_id="transform-1",
            to_node_id="sink-0",
            label="continue",
            mode=RoutingMode.MOVE,
            edge_id="edge-1",
        )

        reason: ConfigGateReason = {"condition": "row['x'] > 0", "result": "true"}
        event = repo.record_routing_event(
            state.state_id,
            "edge-1",
            RoutingMode.MOVE,
            reason=reason,
        )
        assert event.reason_hash is not None

    def test_record_routing_event_with_payload_store(self, tmp_path: Path) -> None:
        """Routing event with payload store persists reason to store."""
        store = FilesystemPayloadStore(tmp_path / "payloads")
        _db, repo, fac, tok = _make_repo_with_token(payload_store=store)
        state = repo.begin_node_state(tok, "transform-1", "run-1", 1, {"a": 1})
        fac.data_flow.register_edge(
            run_id="run-1",
            from_node_id="transform-1",
            to_node_id="sink-0",
            label="continue",
            mode=RoutingMode.MOVE,
            edge_id="edge-1",
        )

        reason: ConfigGateReason = {"condition": "row['x'] > 0", "result": "true"}
        event = repo.record_routing_event(
            state.state_id,
            "edge-1",
            RoutingMode.MOVE,
            reason=reason,
        )
        assert event.reason_ref is not None
        # Verify the payload was stored
        payload_bytes = store.retrieve(event.reason_ref)
        assert b"row['x'] > 0" in payload_bytes

    def test_record_routing_events_multiple(self) -> None:
        """record_routing_events records multiple routes with shared group ID."""
        _db, repo, fac, tok = _make_repo_with_token()
        state = repo.begin_node_state(tok, "transform-1", "run-1", 1, {"a": 1})
        fac.data_flow.register_edge(
            run_id="run-1",
            from_node_id="transform-1",
            to_node_id="sink-0",
            label="path_a",
            mode=RoutingMode.COPY,
            edge_id="edge-a",
        )
        fac.data_flow.register_edge(
            run_id="run-1",
            from_node_id="transform-1",
            to_node_id="sink-0",
            label="path_b",
            mode=RoutingMode.COPY,
            edge_id="edge-b",
        )

        routes = [
            RoutingSpec(edge_id="edge-a", mode=RoutingMode.COPY),
            RoutingSpec(edge_id="edge-b", mode=RoutingMode.COPY),
        ]
        events = repo.record_routing_events(state.state_id, routes)
        assert len(events) == 2
        assert events[0].ordinal == 0
        assert events[1].ordinal == 1
        # All events in a fork share the same routing_group_id
        assert events[0].routing_group_id == events[1].routing_group_id

    def test_record_routing_events_empty_list_returns_empty(self) -> None:
        """record_routing_events with empty routes list returns empty list."""
        _db, repo, _fac, tok = _make_repo_with_token()
        state = repo.begin_node_state(tok, "transform-1", "run-1", 1, {"a": 1})
        events = repo.record_routing_events(state.state_id, [])
        assert events == []


# ---------------------------------------------------------------------------
# register_artifact
# ---------------------------------------------------------------------------


class TestRegisterArtifact:
    """Tests for artifact registration and retrieval."""

    def test_register_artifact_roundtrip(self) -> None:
        """Register an artifact and retrieve it via get_artifacts."""
        _db, repo, _fac, tok = _make_repo_with_token()
        state = repo.begin_node_state(tok, "sink-0", "run-1", 2, {"x": 1})

        artifact = repo.register_artifact(
            run_id="run-1",
            state_id=state.state_id,
            sink_node_id="sink-0",
            artifact_type="csv",
            path="/output/results.csv",
            content_hash="abc123def456",
            size_bytes=1024,
        )
        assert artifact.artifact_type == "csv"
        assert artifact.path_or_uri == "/output/results.csv"
        assert artifact.content_hash == "abc123def456"
        assert artifact.size_bytes == 1024

        # Retrieve via get_artifacts
        artifacts = repo.get_artifacts("run-1")
        assert len(artifacts) == 1
        assert artifacts[0].artifact_id == artifact.artifact_id

    def test_register_artifact_with_idempotency_key(self) -> None:
        """Artifact with idempotency_key stores it for deduplication."""
        _db, repo, _fac, tok = _make_repo_with_token()
        state = repo.begin_node_state(tok, "sink-0", "run-1", 2, {"x": 1})

        artifact = repo.register_artifact(
            run_id="run-1",
            state_id=state.state_id,
            sink_node_id="sink-0",
            artifact_type="json",
            path="/output/data.json",
            content_hash="hash123",
            size_bytes=512,
            idempotency_key="sink-0:row-1:attempt-0",
        )
        assert artifact.idempotency_key == "sink-0:row-1:attempt-0"

    def test_get_artifacts_filtered_by_sink(self) -> None:
        """get_artifacts with sink_node_id filter returns only matching artifacts."""
        _db, repo, fac, tok = _make_repo_with_token()
        # Register a second sink
        fac.data_flow.register_node(
            run_id="run-1",
            plugin_name="json_sink",
            node_type=NodeType.SINK,
            plugin_version="1.0",
            config={},
            node_id="sink-1",
            schema_config=_DYNAMIC_SCHEMA,
        )

        state0 = repo.begin_node_state(tok, "sink-0", "run-1", 2, {"x": 1})
        state1 = repo.begin_node_state(tok, "sink-1", "run-1", 3, {"x": 1}, state_id="state-s1", attempt=1)

        repo.register_artifact("run-1", state0.state_id, "sink-0", "csv", "/out/a.csv", "h1", 100)
        repo.register_artifact("run-1", state1.state_id, "sink-1", "json", "/out/b.json", "h2", 200)

        csv_only = repo.get_artifacts("run-1", sink_node_id="sink-0")
        assert len(csv_only) == 1
        assert csv_only[0].artifact_type == "csv"

        json_only = repo.get_artifacts("run-1", sink_node_id="sink-1")
        assert len(json_only) == 1
        assert json_only[0].artifact_type == "json"

    def test_get_artifacts_empty_run(self) -> None:
        """get_artifacts for a run with no artifacts returns empty list."""
        _db, repo, _fac, _tok = _make_repo_with_token()
        artifacts = repo.get_artifacts("run-1")
        assert artifacts == []


# ---------------------------------------------------------------------------
# Full batch lifecycle (extended)
# ---------------------------------------------------------------------------


class TestBatchLifecycleExtended:
    """Extended batch lifecycle tests: multi-member, status transitions, queries."""

    def test_complete_batch_with_non_terminal_status_raises(self) -> None:
        """complete_batch with non-terminal status raises AuditIntegrityError."""
        _db, repo, _fac, tok = _make_repo_with_token()
        batch = repo.create_batch("run-1", "agg-1")
        repo.add_batch_member(batch.batch_id, tok, 0)

        with pytest.raises(AuditIntegrityError, match="terminal status"):
            repo.complete_batch(batch.batch_id, BatchStatus.DRAFT)

    def test_complete_batch_nonexistent_raises(self) -> None:
        """complete_batch for nonexistent batch raises AuditIntegrityError."""
        _db, repo, _fac, _tok = _make_repo_with_token()
        with pytest.raises(AuditIntegrityError, match="zero rows affected"):
            repo.complete_batch("nonexistent-batch", BatchStatus.COMPLETED)

    def test_complete_batch_already_terminal_raises(self) -> None:
        """Cannot overwrite a terminal batch via complete_batch().

        Regression test for elspeth-28e747cb1e: complete_batch() lacked the
        terminal-state guard that update_batch_status() had, allowing a
        completed batch to be silently rewritten (audit immutability violation).
        """
        _db, repo, _fac, tok = _make_repo_with_token()
        batch = repo.create_batch("run-1", "agg-1")
        repo.add_batch_member(batch.batch_id, tok, 0)
        repo.complete_batch(batch.batch_id, BatchStatus.COMPLETED, trigger_type=TriggerType.COUNT, trigger_reason="c=1")

        with pytest.raises(AuditIntegrityError, match="already terminal"):
            repo.complete_batch(batch.batch_id, BatchStatus.FAILED, trigger_type=TriggerType.TIMEOUT, trigger_reason="t=5")

    def test_update_batch_status_basic(self) -> None:
        """update_batch_status transitions from DRAFT to EXECUTING."""
        _db, repo, _fac, tok = _make_repo_with_token()
        batch = repo.create_batch("run-1", "agg-1")
        repo.add_batch_member(batch.batch_id, tok, 0)

        repo.update_batch_status(batch.batch_id, BatchStatus.EXECUTING)
        updated = repo.get_batch(batch.batch_id)
        assert updated is not None
        assert updated.status == BatchStatus.EXECUTING

    def test_update_batch_status_terminal_raises(self) -> None:
        """Cannot update a batch that's already in terminal status."""
        _db, repo, _fac, tok = _make_repo_with_token()
        batch = repo.create_batch("run-1", "agg-1")
        repo.add_batch_member(batch.batch_id, tok, 0)
        repo.complete_batch(batch.batch_id, BatchStatus.COMPLETED, trigger_type=TriggerType.COUNT, trigger_reason="c=1")

        with pytest.raises(AuditIntegrityError, match="terminal status"):
            repo.update_batch_status(batch.batch_id, BatchStatus.EXECUTING)

    def test_update_batch_status_nonexistent_raises(self) -> None:
        """Updating a nonexistent batch raises AuditIntegrityError."""
        _db, repo, _fac, _tok = _make_repo_with_token()
        with pytest.raises(AuditIntegrityError, match="not found"):
            repo.update_batch_status("nonexistent-batch", BatchStatus.EXECUTING)

    def test_get_batches_with_filters(self) -> None:
        """get_batches filters by status and node_id."""
        _db, repo, _fac, tok = _make_repo_with_token()

        # Create batches for two different nodes
        b1 = repo.create_batch("run-1", "agg-1", batch_id="batch-1")
        repo.add_batch_member(b1.batch_id, tok, 0)
        repo.complete_batch(b1.batch_id, BatchStatus.COMPLETED, trigger_type=TriggerType.COUNT, trigger_reason="c=1")

        b2 = repo.create_batch("run-1", "agg-1", batch_id="batch-2")

        # All batches for agg-1
        all_batches = repo.get_batches("run-1", node_id="agg-1")
        assert len(all_batches) == 2

        # Only completed
        completed = repo.get_batches("run-1", status=BatchStatus.COMPLETED)
        assert len(completed) == 1
        assert completed[0].batch_id == b1.batch_id

        # Only draft
        draft = repo.get_batches("run-1", status=BatchStatus.DRAFT)
        assert len(draft) == 1
        assert draft[0].batch_id == b2.batch_id

    def test_get_incomplete_batches(self) -> None:
        """get_incomplete_batches returns draft, executing, and failed batches."""
        _db, repo, _fac, tok = _make_repo_with_token()

        b_draft = repo.create_batch("run-1", "agg-1", batch_id="batch-draft")
        b_completed = repo.create_batch("run-1", "agg-1", batch_id="batch-done")
        repo.add_batch_member(b_completed.batch_id, tok, 0)
        repo.complete_batch(b_completed.batch_id, BatchStatus.COMPLETED, trigger_type=TriggerType.COUNT, trigger_reason="c=1")

        incomplete = repo.get_incomplete_batches("run-1")
        assert len(incomplete) == 1
        assert incomplete[0].batch_id == b_draft.batch_id

    def test_get_batch_members_ordered_by_ordinal(self) -> None:
        """get_batch_members returns members ordered by ordinal."""
        _db, repo, fac, tok = _make_repo_with_token()
        fac.data_flow.create_token("row-1", token_id="tok-2")
        fac.data_flow.create_token("row-1", token_id="tok-3")

        batch = repo.create_batch("run-1", "agg-1")
        repo.add_batch_member(batch.batch_id, "tok-3", 2)
        repo.add_batch_member(batch.batch_id, tok, 0)
        repo.add_batch_member(batch.batch_id, "tok-2", 1)

        members = repo.get_batch_members(batch.batch_id)
        assert len(members) == 3
        assert [m.ordinal for m in members] == [0, 1, 2]
        assert [m.token_id for m in members] == [tok, "tok-2", "tok-3"]

    def test_get_all_batch_members_for_run(self) -> None:
        """get_all_batch_members_for_run fetches members across all batches."""
        _db, repo, fac, tok = _make_repo_with_token()
        fac.data_flow.create_token("row-1", token_id="tok-2")

        b1 = repo.create_batch("run-1", "agg-1", batch_id="batch-1")
        repo.add_batch_member(b1.batch_id, tok, 0)

        b2 = repo.create_batch("run-1", "agg-1", batch_id="batch-2")
        repo.add_batch_member(b2.batch_id, "tok-2", 0)

        all_members = repo.get_all_batch_members_for_run("run-1")
        assert len(all_members) == 2

    def test_get_batch_returns_none_for_missing(self) -> None:
        """get_batch returns None for a nonexistent batch_id."""
        _db, repo, _fac, _tok = _make_repo_with_token()
        assert repo.get_batch("nonexistent") is None

    def test_create_batch_with_explicit_attempt(self) -> None:
        """create_batch with explicit attempt number uses it."""
        _db, repo, _fac, _tok = _make_repo_with_token()
        batch = repo.create_batch("run-1", "agg-1", attempt=5)
        assert batch.attempt == 5


# ---------------------------------------------------------------------------
# Operations queries
# ---------------------------------------------------------------------------


class TestOperationQueries:
    """Tests for operation query methods."""

    def test_get_operations_for_run(self) -> None:
        """get_operations_for_run returns all operations ordered by started_at."""
        _db, repo, _fac, _tok = _make_repo_with_token()
        op1 = repo.begin_operation("run-1", "source-0", "source_load")
        op2 = repo.begin_operation("run-1", "sink-0", "sink_write")

        ops = repo.get_operations_for_run("run-1")
        assert len(ops) == 2
        assert ops[0].operation_id == op1.operation_id
        assert ops[1].operation_id == op2.operation_id

    def test_get_all_operation_calls_for_run(self) -> None:
        """get_all_operation_calls_for_run returns all operation-parented calls."""
        _db, repo, _fac, _tok = _make_repo_with_token()
        op1 = repo.begin_operation("run-1", "source-0", "source_load")
        op2 = repo.begin_operation("run-1", "sink-0", "sink_write")

        repo.record_operation_call(
            op1.operation_id,
            CallType.HTTP,
            CallStatus.SUCCESS,
            RawCallPayload({"url": "a"}),
            RawCallPayload({"resp": "a"}),
        )
        repo.record_operation_call(
            op2.operation_id,
            CallType.HTTP,
            CallStatus.SUCCESS,
            RawCallPayload({"url": "b"}),
            RawCallPayload({"resp": "b"}),
        )

        all_calls = repo.get_all_operation_calls_for_run("run-1")
        assert len(all_calls) == 2

    def test_get_operation_returns_none_for_missing(self) -> None:
        """get_operation returns None for nonexistent operation_id."""
        _db, repo, _fac, _tok = _make_repo_with_token()
        assert repo.get_operation("nonexistent-op") is None


# ---------------------------------------------------------------------------
# Call recording with payload store
# ---------------------------------------------------------------------------


class TestCallRecordingWithPayloadStore:
    """Tests for record_call with payload store auto-persistence."""

    def test_record_call_auto_persists_to_payload_store(self, tmp_path: Path) -> None:
        """record_call auto-persists request and response to payload store."""
        store = FilesystemPayloadStore(tmp_path / "payloads")
        _db, repo, _fac, tok = _make_repo_with_token(payload_store=store)
        state = repo.begin_node_state(tok, "transform-1", "run-1", 1, {"a": 1})
        idx = repo.allocate_call_index(state.state_id)

        call = repo.record_call(
            state.state_id,
            idx,
            CallType.LLM,
            CallStatus.SUCCESS,
            RawCallPayload({"prompt": "test"}),
            RawCallPayload({"response": "classified"}),
        )
        assert call.request_ref is not None
        assert call.response_ref is not None

        # Verify stored data is retrievable
        req_bytes = store.retrieve(call.request_ref)
        assert b"prompt" in req_bytes
        resp_bytes = store.retrieve(call.response_ref)
        assert b"classified" in resp_bytes

    def test_record_call_explicit_refs_skip_auto_persist(self, tmp_path: Path) -> None:
        """record_call with explicit request_ref/response_ref skips auto-persist."""
        store = FilesystemPayloadStore(tmp_path / "payloads")
        _db, repo, _fac, tok = _make_repo_with_token(payload_store=store)
        state = repo.begin_node_state(tok, "transform-1", "run-1", 1, {"a": 1})
        idx = repo.allocate_call_index(state.state_id)

        call = repo.record_call(
            state.state_id,
            idx,
            CallType.LLM,
            CallStatus.SUCCESS,
            RawCallPayload({"prompt": "test"}),
            RawCallPayload({"response": "result"}),
            request_ref="existing-ref-123",
            response_ref="existing-ref-456",
        )
        # Should use the provided refs, not auto-generate
        assert call.request_ref == "existing-ref-123"
        assert call.response_ref == "existing-ref-456"

    def test_record_call_error_without_response(self) -> None:
        """record_call with error status and no response data."""
        _db, repo, _fac, tok = _make_repo_with_token()
        state = repo.begin_node_state(tok, "transform-1", "run-1", 1, {"a": 1})
        idx = repo.allocate_call_index(state.state_id)

        call = repo.record_call(
            state.state_id,
            idx,
            CallType.LLM,
            CallStatus.ERROR,
            RawCallPayload({"prompt": "test"}),
            error=RawCallPayload({"error": "timeout"}),
        )
        assert call.status == CallStatus.ERROR
        assert call.response_hash is None
        assert call.error_json is not None

    def test_record_call_failed_insert_does_not_store_payloads(self) -> None:
        """Failed call inserts must not leak request/response payload blobs."""
        store = _TrackingPayloadStore()
        _db, repo, _fac, _tok = _make_repo_with_token(payload_store=store)

        with pytest.raises(LandscapeRecordError, match="FOREIGN KEY constraint failed"):
            repo.record_call(
                "missing-state",
                0,
                CallType.LLM,
                CallStatus.SUCCESS,
                RawCallPayload({"prompt": "bad"}),
                RawCallPayload({"response": "bad"}),
            )

        assert store.store_calls == []

    def test_record_operation_call_failed_insert_does_not_store_payloads(self) -> None:
        """Failed operation-call inserts must not leak request/response payload blobs."""
        store = _TrackingPayloadStore()
        _db, repo, _fac, _tok = _make_repo_with_token(payload_store=store)

        with pytest.raises(LandscapeRecordError, match="FOREIGN KEY constraint failed"):
            repo.record_operation_call(
                "missing-operation",
                CallType.HTTP,
                CallStatus.SUCCESS,
                RawCallPayload({"url": "https://example.com"}),
                RawCallPayload({"status": 200}),
            )

        assert store.store_calls == []

    def test_record_routing_event_failed_insert_does_not_store_reason(self) -> None:
        """Failed routing-event inserts must not leak reason payloads."""
        store = _TrackingPayloadStore()
        _db, repo, _fac, tok = _make_repo_with_token(payload_store=store)
        state = repo.begin_node_state(tok, "transform-1", "run-1", 1, {"a": 1})

        with pytest.raises(LandscapeRecordError, match="FOREIGN KEY constraint failed"):
            repo.record_routing_event(
                state.state_id,
                "missing-edge",
                RoutingMode.MOVE,
                reason={"condition": "x", "result": "true"},
            )

        assert store.store_calls == []

    def test_record_routing_events_failed_insert_does_not_store_shared_reason(self) -> None:
        """Failed multi-routing inserts must not materialize the shared reason blob."""
        store = _TrackingPayloadStore()
        _db, repo, fac, tok = _make_repo_with_token(payload_store=store)
        state = repo.begin_node_state(tok, "transform-1", "run-1", 1, {"a": 1})
        fac.data_flow.register_edge(
            run_id="run-1",
            from_node_id="transform-1",
            to_node_id="sink-0",
            label="path-a",
            mode=RoutingMode.COPY,
            edge_id="edge-a",
        )

        with pytest.raises((LandscapeRecordError, SQLAlchemyError), match="FOREIGN KEY constraint failed"):
            repo.record_routing_events(
                state.state_id,
                [
                    RoutingSpec(edge_id="edge-a", mode=RoutingMode.COPY),
                    RoutingSpec(edge_id="missing-edge", mode=RoutingMode.COPY),
                ],
                reason={"condition": "fork", "result": "true"},
            )

        assert store.store_calls == []


# ---------------------------------------------------------------------------
# M4: Delegation signature alignment test
# ---------------------------------------------------------------------------


class TestDelegationSignatureAlignment:
    """Verify RecorderFactory.execution exposes all ExecutionRepository methods.

    This test verifies that every method on ExecutionRepository is accessible
    through the factory's execution property, ensuring the factory doesn't
    drift from the repository.
    """

    # Methods expected on RecorderFactory.execution (ExecutionRepository)
    _DELEGATED_METHODS: ClassVar[list[str]] = [
        "begin_node_state",
        "complete_node_state",
        "get_node_state",
        "record_routing_event",
        "record_routing_events",
        "allocate_call_index",
        "record_call",
        "begin_operation",
        "complete_operation",
        "allocate_operation_call_index",
        "record_operation_call",
        "get_operation",
        "get_operation_calls",
        "get_operations_for_run",
        "get_all_operation_calls_for_run",
        "find_call_by_request_hash",
        "get_call_response_data",
        "create_batch",
        "add_batch_member",
        "update_batch_status",
        "complete_batch",
        "get_batch",
        "get_batches",
        "get_incomplete_batches",
        "get_batch_members",
        "get_all_batch_members_for_run",
        "retry_batch",
        "register_artifact",
        "get_artifacts",
    ]

    @pytest.mark.parametrize("method_name", _DELEGATED_METHODS)
    def test_signature_alignment(self, method_name: str) -> None:
        """Method must exist on RecorderFactory.execution with correct signature."""
        factory = make_factory()
        factory_method = getattr(factory.execution, method_name)
        repo_method = getattr(ExecutionRepository, method_name)

        factory_sig = inspect.signature(factory_method)
        repo_sig = inspect.signature(repo_method)

        factory_params = [(name, p.kind, p.default) for name, p in factory_sig.parameters.items() if name != "self"]
        repo_params = [(name, p.kind, p.default) for name, p in repo_sig.parameters.items() if name != "self"]

        assert factory_params == repo_params, (
            f"Signature mismatch for {method_name}:\n  Factory:  {factory_params}\n  Repo:     {repo_params}"
        )


class TestResolvedPromptTemplateHashAnchor:
    """A bad resolved_prompt_template_hash must leave zero `calls` rows.

    Regression for elspeth-a94e626a36: the hash anchor was inserted into `calls`
    before Call.__post_init__ validated it, so a non-LLM or malformed hash
    committed an audit row and only then raised — a Tier-1 violation. The
    invariant is now validated BEFORE any insert at both write sites.
    """

    @staticmethod
    def _calls_row_count(db: LandscapeDB) -> int:
        from sqlalchemy import func, select

        from elspeth.core.landscape.schema import calls_table

        with db.connection() as conn:
            return int(conn.execute(select(func.count()).select_from(calls_table)).scalar_one())

    def test_state_call_non_llm_hash_leaves_no_row(self) -> None:
        db, repo, _fac, tok = _make_repo_with_token()
        state = repo.begin_node_state(tok, "transform-1", "run-1", 1, {"a": 1})
        idx = repo.allocate_call_index(state.state_id)

        # Non-LLM call carrying a (well-formed) hash violates the LLM-only rule.
        with pytest.raises(ValueError, match="only for CallType"):
            repo.record_call(
                state.state_id,
                idx,
                CallType.HTTP,
                CallStatus.SUCCESS,
                RawCallPayload({"prompt": "hello"}),
                resolved_prompt_template_hash="a" * 64,
            )
        assert self._calls_row_count(db) == 0

    def test_state_call_malformed_hash_leaves_no_row(self) -> None:
        db, repo, _fac, tok = _make_repo_with_token()
        state = repo.begin_node_state(tok, "transform-1", "run-1", 1, {"a": 1})
        idx = repo.allocate_call_index(state.state_id)

        with pytest.raises(ValueError, match="64-character lowercase hex"):
            repo.record_call(
                state.state_id,
                idx,
                CallType.LLM,
                CallStatus.SUCCESS,
                RawCallPayload({"prompt": "hello"}),
                resolved_prompt_template_hash="not-a-valid-sha256",
            )
        assert self._calls_row_count(db) == 0

    def test_operation_call_non_llm_hash_leaves_no_row(self) -> None:
        db, repo, _fac, _tok = _make_repo_with_token()
        op = repo.begin_operation("run-1", "source-0", "source_load")

        with pytest.raises(ValueError, match="only for CallType"):
            repo.record_operation_call(
                op.operation_id,
                CallType.HTTP,
                CallStatus.SUCCESS,
                RawCallPayload({"prompt": "hello"}),
                resolved_prompt_template_hash="a" * 64,
            )
        assert self._calls_row_count(db) == 0

    def test_operation_call_malformed_hash_leaves_no_row(self) -> None:
        db, repo, _fac, _tok = _make_repo_with_token()
        op = repo.begin_operation("run-1", "source-0", "source_load")

        with pytest.raises(ValueError, match="64-character lowercase hex"):
            repo.record_operation_call(
                op.operation_id,
                CallType.LLM,
                CallStatus.SUCCESS,
                RawCallPayload({"prompt": "hello"}),
                resolved_prompt_template_hash="ABC",
            )
        assert self._calls_row_count(db) == 0

    def test_valid_llm_hash_still_records(self) -> None:
        """A well-formed LLM hash records exactly one row (fix must not over-reject)."""
        db, repo, _fac, tok = _make_repo_with_token()
        state = repo.begin_node_state(tok, "transform-1", "run-1", 1, {"a": 1})
        idx = repo.allocate_call_index(state.state_id)

        call = repo.record_call(
            state.state_id,
            idx,
            CallType.LLM,
            CallStatus.SUCCESS,
            RawCallPayload({"prompt": "hello"}),
            resolved_prompt_template_hash="a" * 64,
        )
        assert call.resolved_prompt_template_hash == "a" * 64
        assert self._calls_row_count(db) == 1
