"""Tests for RecorderFactory node state operations."""

from __future__ import annotations

from elspeth.contracts import NodeStateStatus, NodeType
from elspeth.contracts.audit import NodeStateCompleted
from elspeth.contracts.errors import AuditIntegrityError
from elspeth.contracts.schema import SchemaConfig

# Dynamic schema for tests that don't care about specific fields
DYNAMIC_SCHEMA = SchemaConfig.from_dict({"mode": "observed"})


class TestRecorderFactoryNodeStates:
    """Node state recording (what happened at each node)."""

    def test_begin_node_state(self) -> None:
        from elspeth.contracts import NodeStateStatus
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.factory import RecorderFactory

        db = LandscapeDB.in_memory()
        factory = RecorderFactory(db)
        run = factory.run_lifecycle.begin_run(config={}, canonical_version="v1")
        source = factory.data_flow.register_node(
            run_id=run.run_id,
            plugin_name="source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        row = factory.data_flow.create_row(
            run_id=run.run_id,
            source_node_id=source.node_id,
            row_index=0,
            data={},
        )
        token = factory.data_flow.create_token(row_id=row.row_id)

        state = factory.execution.begin_node_state(
            token_id=token.token_id,
            node_id=source.node_id,
            run_id=run.run_id,
            step_index=0,
            input_data={"value": 42},
        )

        assert state.state_id is not None
        assert state.status == NodeStateStatus.OPEN
        assert state.input_hash is not None

    def test_node_state_hash_correctness(self) -> None:
        """P1: Verify input/output hashes match stable_hash (not just non-NULL).

        Hash correctness is the audit integrity anchor. Tests must validate
        the actual hash values, not just check for existence.
        """
        from elspeth.contracts import NodeStateStatus
        from elspeth.core.canonical import stable_hash
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.factory import RecorderFactory

        db = LandscapeDB.in_memory()
        factory = RecorderFactory(db)
        run = factory.run_lifecycle.begin_run(config={}, canonical_version="v1")
        node = factory.data_flow.register_node(
            run_id=run.run_id,
            plugin_name="transform",
            node_type=NodeType.TRANSFORM,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        row = factory.data_flow.create_row(
            run_id=run.run_id,
            source_node_id=node.node_id,
            row_index=0,
            data={},
        )
        token = factory.data_flow.create_token(row_id=row.row_id)

        input_data = {"x": 1, "y": 2}
        output_data = {"x": 1, "y": 2, "z": 3}

        state = factory.execution.begin_node_state(
            token_id=token.token_id,
            node_id=node.node_id,
            run_id=run.run_id,
            step_index=0,
            input_data=input_data,
        )

        # P1: Verify input_hash matches expected
        expected_input_hash = stable_hash(input_data)
        assert state.input_hash == expected_input_hash, f"input_hash mismatch: expected {expected_input_hash}, got {state.input_hash}"

        completed = factory.execution.complete_node_state(
            state_id=state.state_id,
            status=NodeStateStatus.COMPLETED,
            output_data=output_data,
            duration_ms=10.5,
        )

        # P1: Verify output_hash matches expected
        assert completed.status == NodeStateStatus.COMPLETED
        expected_output_hash = stable_hash(output_data)
        assert completed.output_hash == expected_output_hash, (
            f"output_hash mismatch: expected {expected_output_hash}, got {completed.output_hash}"
        )

    def test_complete_node_state_success(self) -> None:
        from elspeth.contracts import NodeStateStatus
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.factory import RecorderFactory

        db = LandscapeDB.in_memory()
        factory = RecorderFactory(db)
        run = factory.run_lifecycle.begin_run(config={}, canonical_version="v1")
        node = factory.data_flow.register_node(
            run_id=run.run_id,
            plugin_name="transform",
            node_type=NodeType.TRANSFORM,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        row = factory.data_flow.create_row(
            run_id=run.run_id,
            source_node_id=node.node_id,
            row_index=0,
            data={},
        )
        token = factory.data_flow.create_token(row_id=row.row_id)
        state = factory.execution.begin_node_state(
            token_id=token.token_id,
            node_id=node.node_id,
            run_id=run.run_id,
            step_index=0,
            input_data={"x": 1},
        )

        completed = factory.execution.complete_node_state(
            state_id=state.state_id,
            status=NodeStateStatus.COMPLETED,
            output_data={"x": 1, "y": 2},
            duration_ms=10.5,
        )

        assert completed.status == NodeStateStatus.COMPLETED
        assert completed.output_hash is not None
        assert completed.duration_ms == 10.5
        assert completed.completed_at is not None

    def test_complete_node_state_failed(self) -> None:
        from elspeth.contracts import NodeStateStatus
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.factory import RecorderFactory

        db = LandscapeDB.in_memory()
        factory = RecorderFactory(db)
        run = factory.run_lifecycle.begin_run(config={}, canonical_version="v1")
        node = factory.data_flow.register_node(
            run_id=run.run_id,
            plugin_name="transform",
            node_type=NodeType.TRANSFORM,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        row = factory.data_flow.create_row(
            run_id=run.run_id,
            source_node_id=node.node_id,
            row_index=0,
            data={},
        )
        token = factory.data_flow.create_token(row_id=row.row_id)
        state = factory.execution.begin_node_state(
            token_id=token.token_id,
            node_id=node.node_id,
            run_id=run.run_id,
            step_index=0,
            input_data={},
        )

        completed = factory.execution.complete_node_state(
            state_id=state.state_id,
            status=NodeStateStatus.FAILED,
            error={"reason": "test_error", "message": "Validation failed"},
            duration_ms=5.0,
        )

        assert completed.status == NodeStateStatus.FAILED
        assert completed.error_json is not None
        assert "Validation failed" in completed.error_json

    def test_complete_node_state_failed_with_execution_error(self) -> None:
        """ExecutionError frozen dataclass serializes via to_dict() through complete_node_state().

        Critical: The isinstance dispatch at _node_state_recording.py:199 calls
        ExecutionError.to_dict() which maps exception_type -> "type" for hash
        stability.  This test verifies the full path:
          executor creates ExecutionError -> complete_node_state() -> isinstance
          dispatches -> to_dict() serializes with "type" key -> canonical_json()
          -> stored in DB -> readable with correct schema.

        Bug: 3.3-fixes #3 (elspeth-rapid-1168a1)
        """
        import json

        from elspeth.contracts import ExecutionError, NodeStateStatus
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.factory import RecorderFactory

        db = LandscapeDB.in_memory()
        factory = RecorderFactory(db)
        run = factory.run_lifecycle.begin_run(config={}, canonical_version="v1")
        node = factory.data_flow.register_node(
            run_id=run.run_id,
            plugin_name="transform",
            node_type=NodeType.TRANSFORM,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        row = factory.data_flow.create_row(
            run_id=run.run_id,
            source_node_id=node.node_id,
            row_index=0,
            data={},
        )
        token = factory.data_flow.create_token(row_id=row.row_id)
        state = factory.execution.begin_node_state(
            token_id=token.token_id,
            node_id=node.node_id,
            run_id=run.run_id,
            step_index=0,
            input_data={},
        )

        error = ExecutionError(
            exception="division by zero",
            exception_type="ZeroDivisionError",
            traceback="Traceback (most recent call last):\n  File ...",
            phase="process",
        )

        completed = factory.execution.complete_node_state(
            state_id=state.state_id,
            status=NodeStateStatus.FAILED,
            error=error,
            duration_ms=3.0,
        )

        # Verify the error was stored
        assert completed.status == NodeStateStatus.FAILED
        assert completed.error_json is not None

        # Critical assertion: the stored JSON uses "type" (from to_dict()),
        # NOT "exception_type" (the dataclass field name).
        stored = json.loads(completed.error_json)
        assert "type" in stored, (
            f"error_json must contain 'type' key (from ExecutionError.to_dict()), not 'exception_type'. Got keys: {list(stored.keys())}"
        )
        assert "exception_type" not in stored, (
            "error_json must NOT contain 'exception_type' — ExecutionError.to_dict() should map it to 'type'"
        )
        assert stored["type"] == "ZeroDivisionError"
        assert stored["exception"] == "division by zero"
        assert stored["traceback"] == "Traceback (most recent call last):\n  File ..."
        assert stored["phase"] == "process"

    def test_complete_node_state_failed_execution_error_omits_none_fields(self) -> None:
        """ExecutionError.to_dict() omits None-valued optional fields for hash stability.

        Bug: 3.3-fixes #3 (elspeth-rapid-1168a1)
        """
        import json

        from elspeth.contracts import ExecutionError, NodeStateStatus
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.factory import RecorderFactory

        db = LandscapeDB.in_memory()
        factory = RecorderFactory(db)
        run = factory.run_lifecycle.begin_run(config={}, canonical_version="v1")
        node = factory.data_flow.register_node(
            run_id=run.run_id,
            plugin_name="gate",
            node_type=NodeType.GATE,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        row = factory.data_flow.create_row(
            run_id=run.run_id,
            source_node_id=node.node_id,
            row_index=0,
            data={},
        )
        token = factory.data_flow.create_token(row_id=row.row_id)
        state = factory.execution.begin_node_state(
            token_id=token.token_id,
            node_id=node.node_id,
            run_id=run.run_id,
            step_index=0,
            input_data={},
        )

        # Minimal error — no traceback, no phase
        error = ExecutionError(
            exception="some error",
            exception_type="RuntimeError",
        )

        completed = factory.execution.complete_node_state(
            state_id=state.state_id,
            status=NodeStateStatus.FAILED,
            error=error,
            duration_ms=1.0,
        )

        assert completed.error_json is not None
        stored = json.loads(completed.error_json)
        # Only required fields present — None-valued optional fields omitted
        assert set(stored.keys()) == {"exception", "type"}
        assert "traceback" not in stored
        assert "phase" not in stored

    def test_complete_node_state_with_empty_output(self) -> None:
        """Empty dict output is valid and must produce non-NULL output_hash.

        Bug: P1-2026-01-19-complete-node-state-empty-output-hash
        """
        from elspeth.contracts import NodeStateStatus
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.factory import RecorderFactory

        db = LandscapeDB.in_memory()
        factory = RecorderFactory(db)
        run = factory.run_lifecycle.begin_run(config={}, canonical_version="v1")
        node = factory.data_flow.register_node(
            run_id=run.run_id,
            plugin_name="transform",
            node_type=NodeType.TRANSFORM,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        row = factory.data_flow.create_row(
            run_id=run.run_id,
            source_node_id=node.node_id,
            row_index=0,
            data={},
        )
        token = factory.data_flow.create_token(row_id=row.row_id)
        state = factory.execution.begin_node_state(
            token_id=token.token_id,
            node_id=node.node_id,
            run_id=run.run_id,
            step_index=0,
            input_data={},
        )

        # Empty output_data={} should succeed, not crash
        completed = factory.execution.complete_node_state(
            state_id=state.state_id,
            status=NodeStateStatus.COMPLETED,
            output_data={},  # Empty dict is valid output
            duration_ms=1.0,
        )

        assert completed.status == NodeStateStatus.COMPLETED
        assert completed.output_hash is not None  # Must have non-NULL hash

    def test_complete_node_state_with_empty_error(self) -> None:
        """Empty dict error payload is recorded, not dropped.

        Bug: P1-2026-01-19-complete-node-state-empty-output-hash
        """
        from elspeth.contracts import NodeStateStatus
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.factory import RecorderFactory

        db = LandscapeDB.in_memory()
        factory = RecorderFactory(db)
        run = factory.run_lifecycle.begin_run(config={}, canonical_version="v1")
        node = factory.data_flow.register_node(
            run_id=run.run_id,
            plugin_name="transform",
            node_type=NodeType.TRANSFORM,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        row = factory.data_flow.create_row(
            run_id=run.run_id,
            source_node_id=node.node_id,
            row_index=0,
            data={},
        )
        token = factory.data_flow.create_token(row_id=row.row_id)
        state = factory.execution.begin_node_state(
            token_id=token.token_id,
            node_id=node.node_id,
            run_id=run.run_id,
            step_index=0,
            input_data={},
        )

        # Empty error={} should be serialized, not dropped
        completed = factory.execution.complete_node_state(  # type: ignore[call-overload]  # Empty dict tests serialization
            state_id=state.state_id,
            status=NodeStateStatus.FAILED,
            error={},
            duration_ms=1.0,
        )

        assert completed.status == NodeStateStatus.FAILED
        assert completed.error_json == "{}"  # Empty dict serializes to "{}"

    def test_retry_increments_attempt(self) -> None:
        from elspeth.contracts import NodeStateStatus
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.factory import RecorderFactory

        db = LandscapeDB.in_memory()
        factory = RecorderFactory(db)
        run = factory.run_lifecycle.begin_run(config={}, canonical_version="v1")
        node = factory.data_flow.register_node(
            run_id=run.run_id,
            plugin_name="transform",
            node_type=NodeType.TRANSFORM,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        row = factory.data_flow.create_row(
            run_id=run.run_id,
            source_node_id=node.node_id,
            row_index=0,
            data={},
        )
        token = factory.data_flow.create_token(row_id=row.row_id)

        # First attempt fails
        state1 = factory.execution.begin_node_state(
            token_id=token.token_id,
            node_id=node.node_id,
            run_id=run.run_id,
            step_index=0,
            input_data={},
            attempt=0,
        )
        factory.execution.complete_node_state(state1.state_id, status=NodeStateStatus.FAILED, error={}, duration_ms=1.0)  # type: ignore[call-overload]  # Empty dict tests serialization

        # Second attempt
        state2 = factory.execution.begin_node_state(
            token_id=token.token_id,
            node_id=node.node_id,
            run_id=run.run_id,
            step_index=0,
            input_data={},
            attempt=1,
        )

        assert state2.attempt == 1


class TestNodeStateIntegrityValidation:
    """Regression tests for Tier 1 audit integrity validation.

    Bug: P2-2026-01-19-node-state-terminal-completed-at-not-validated
    Terminal node states (COMPLETED, FAILED) must have non-NULL completed_at.
    Reading corrupted audit data should crash per Data Manifesto Tier 1 rules.
    """

    def test_completed_state_with_null_completed_at_raises(self) -> None:
        """COMPLETED state with NULL completed_at raises integrity violation.

        Per Data Manifesto: "Bad data in audit trail = crash immediately"
        """
        import pytest
        from sqlalchemy import text

        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.factory import RecorderFactory

        db = LandscapeDB.in_memory()
        factory = RecorderFactory(db)

        # Create valid infrastructure
        run = factory.run_lifecycle.begin_run(config={}, canonical_version="v1")
        node = factory.data_flow.register_node(
            run_id=run.run_id,
            plugin_name="test_source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        row_record = factory.data_flow.create_row(
            run_id=run.run_id,
            source_node_id=node.node_id,
            row_index=0,
            data={"test": "value"},
        )
        token = factory.data_flow.create_token(row_id=row_record.row_id)

        # Create a completed state normally
        state = factory.execution.begin_node_state(
            token_id=token.token_id,
            node_id=node.node_id,
            run_id=run.run_id,
            step_index=0,
            input_data={"test": "data"},
        )
        factory.execution.complete_node_state(
            state_id=state.state_id,
            status=NodeStateStatus.COMPLETED,
            output_data={"result": "ok"},
            duration_ms=10.0,
        )

        # Verify it works normally
        retrieved = factory.query.get_node_states_for_token(token.token_id)
        assert len(retrieved) == 1

        # Now corrupt the database - set completed_at to NULL
        with db.connection() as conn:
            conn.execute(
                text("UPDATE node_states SET completed_at = NULL WHERE state_id = :sid"),
                {"sid": state.state_id},
            )
            conn.commit()

        # Reading corrupted data should crash (Tier 1 rule)
        with pytest.raises(AuditIntegrityError, match=r"NULL completed_at.*audit integrity violation"):
            factory.query.get_node_states_for_token(token.token_id)

    def test_failed_state_with_null_completed_at_raises(self) -> None:
        """FAILED state with NULL completed_at raises integrity violation.

        Per Data Manifesto: "Bad data in audit trail = crash immediately"
        """
        import pytest
        from sqlalchemy import text

        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.factory import RecorderFactory

        db = LandscapeDB.in_memory()
        factory = RecorderFactory(db)

        # Create valid infrastructure
        run = factory.run_lifecycle.begin_run(config={}, canonical_version="v1")
        node = factory.data_flow.register_node(
            run_id=run.run_id,
            plugin_name="test_source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        row_record = factory.data_flow.create_row(
            run_id=run.run_id,
            source_node_id=node.node_id,
            row_index=0,
            data={"test": "value"},
        )
        token = factory.data_flow.create_token(row_id=row_record.row_id)

        # Create a failed state normally
        state = factory.execution.begin_node_state(
            token_id=token.token_id,
            node_id=node.node_id,
            run_id=run.run_id,
            step_index=0,
            input_data={"test": "data"},
        )
        factory.execution.complete_node_state(
            state_id=state.state_id,
            status=NodeStateStatus.FAILED,
            error={"reason": "test_error", "message": "Something went wrong"},
            duration_ms=5.0,
        )

        # Verify it works normally
        retrieved = factory.query.get_node_states_for_token(token.token_id)
        assert len(retrieved) == 1

        # Now corrupt the database - set completed_at to NULL
        with db.connection() as conn:
            conn.execute(
                text("UPDATE node_states SET completed_at = NULL WHERE state_id = :sid"),
                {"sid": state.state_id},
            )
            conn.commit()

        # Reading corrupted data should crash (Tier 1 rule)
        with pytest.raises(AuditIntegrityError, match=r"NULL completed_at.*audit integrity violation"):
            factory.query.get_node_states_for_token(token.token_id)


class TestNodeStateOrderingWithRetries:
    """Regression tests for P2-2026-01-19-node-state-ordering-missing-attempt.

    Node states must be ordered by (step_index, attempt) for deterministic
    output, especially when retries exist.
    """

    def test_get_node_states_orders_by_step_index_and_attempt(self) -> None:
        """Node states are returned ordered by (step_index, attempt).

        Bug: Query only ordered by step_index, leaving attempt ordering
        undefined across database backends. This caused non-deterministic
        output for retries and could break signed exports.

        Fix: ORDER BY (step_index, attempt) for deterministic ordering.
        """
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.factory import RecorderFactory

        db = LandscapeDB.in_memory()
        factory = RecorderFactory(db)

        run = factory.run_lifecycle.begin_run(config={}, canonical_version="v1")
        node1 = factory.data_flow.register_node(
            run_id=run.run_id,
            plugin_name="test_transform_1",
            node_type=NodeType.TRANSFORM,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        node2 = factory.data_flow.register_node(
            run_id=run.run_id,
            plugin_name="test_transform_2",
            node_type=NodeType.TRANSFORM,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        row_record = factory.data_flow.create_row(
            run_id=run.run_id,
            source_node_id=node1.node_id,
            row_index=0,
            data={"test": "value"},
        )
        token = factory.data_flow.create_token(row_id=row_record.row_id)

        # Create states at step 0 with multiple attempts (simulating retries)
        # Insert OUT OF ORDER to test that ordering is enforced by the query
        state_0_attempt_1 = factory.execution.begin_node_state(
            token_id=token.token_id,
            node_id=node1.node_id,
            run_id=run.run_id,
            step_index=0,
            attempt=1,  # Second attempt first!
            input_data={"test": "data"},
        )
        factory.execution.complete_node_state(
            state_id=state_0_attempt_1.state_id,
            status=NodeStateStatus.FAILED,
            error={"reason": "test_error", "message": "First failure"},
            duration_ms=10.0,
        )

        state_0_attempt_0 = factory.execution.begin_node_state(
            token_id=token.token_id,
            node_id=node1.node_id,
            run_id=run.run_id,
            step_index=0,
            attempt=0,  # First attempt second!
            input_data={"test": "data"},
        )
        factory.execution.complete_node_state(
            state_id=state_0_attempt_0.state_id,
            status=NodeStateStatus.COMPLETED,
            output_data={"result": "ok"},
            duration_ms=5.0,
        )

        # Create a state at step 1 using a different node (different step in pipeline)
        state_1_attempt_0 = factory.execution.begin_node_state(
            token_id=token.token_id,
            node_id=node2.node_id,  # Different node for step 1
            run_id=run.run_id,
            step_index=1,
            attempt=0,
            input_data={"test": "data2"},
        )
        factory.execution.complete_node_state(
            state_id=state_1_attempt_0.state_id,
            status=NodeStateStatus.COMPLETED,
            output_data={"result": "ok2"},
            duration_ms=3.0,
        )

        # REGRESSION CHECK: Verify ordering is (step_index, attempt)
        states = factory.query.get_node_states_for_token(token.token_id)
        assert len(states) == 3

        # Verify order: step 0 attempt 0, step 0 attempt 1, step 1 attempt 0
        assert states[0].step_index == 0
        assert states[0].attempt == 0
        assert states[1].step_index == 0
        assert states[1].attempt == 1
        assert states[2].step_index == 1
        assert states[2].attempt == 0

        # Verify the state IDs match expected order
        assert states[0].state_id == state_0_attempt_0.state_id
        assert states[1].state_id == state_0_attempt_1.state_id
        assert states[2].state_id == state_1_attempt_0.state_id


class TestContextAfterRoundTrip:
    """DB round-trip tests for typed context_after metadata.

    Verifies: construct → .to_dict() → canonical_json() → DB write → DB read → JSON parse
    """

    def test_coalesce_metadata_round_trip(self) -> None:
        """CoalesceMetadata survives full DB round-trip."""
        import json

        from elspeth.contracts.coalesce_enums import CoalescePolicy, MergeStrategy
        from elspeth.contracts.coalesce_metadata import ArrivalOrderEntry, CoalesceMetadata
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.factory import RecorderFactory

        db = LandscapeDB.in_memory()
        factory = RecorderFactory(db)
        run = factory.run_lifecycle.begin_run(config={}, canonical_version="v1")
        node = factory.data_flow.register_node(
            run_id=run.run_id,
            plugin_name="coalesce",
            node_type=NodeType.COALESCE,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        row = factory.data_flow.create_row(
            run_id=run.run_id,
            source_node_id=node.node_id,
            row_index=0,
            data={},
        )
        token = factory.data_flow.create_token(row_id=row.row_id)

        metadata = CoalesceMetadata.for_merge(
            policy=CoalescePolicy.REQUIRE_ALL,
            merge_strategy=MergeStrategy.UNION,
            expected_branches=["a", "b"],
            branches_arrived=["a", "b"],
            branches_lost={},
            arrival_order=[
                ArrivalOrderEntry(branch="a", arrival_offset_ms=0.0),
                ArrivalOrderEntry(branch="b", arrival_offset_ms=50.0),
            ],
            wait_duration_ms=50.0,
        )

        state = factory.execution.begin_node_state(
            token_id=token.token_id,
            node_id=node.node_id,
            run_id=run.run_id,
            step_index=0,
            input_data={},
        )
        factory.execution.complete_node_state(
            state_id=state.state_id,
            status=NodeStateStatus.COMPLETED,
            output_data={"merged": True},
            duration_ms=50.0,
            context_after=metadata,
        )

        fetched = factory.execution.get_node_state(state.state_id)
        assert fetched is not None
        assert isinstance(fetched, NodeStateCompleted)
        assert fetched.context_after_json is not None
        assert json.loads(fetched.context_after_json) == metadata.to_dict()

    def test_pool_execution_context_round_trip(self) -> None:
        """PoolExecutionContext survives full DB round-trip."""
        import json

        from elspeth.contracts.node_state_context import (
            PoolConfigSnapshot,
            PoolExecutionContext,
            PoolStatsSnapshot,
            QueryOrderEntry,
        )
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.factory import RecorderFactory

        db = LandscapeDB.in_memory()
        factory = RecorderFactory(db)
        run = factory.run_lifecycle.begin_run(config={}, canonical_version="v1")
        node = factory.data_flow.register_node(
            run_id=run.run_id,
            plugin_name="transform",
            node_type=NodeType.TRANSFORM,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        row = factory.data_flow.create_row(
            run_id=run.run_id,
            source_node_id=node.node_id,
            row_index=0,
            data={},
        )
        token = factory.data_flow.create_token(row_id=row.row_id)

        ctx = PoolExecutionContext(
            pool_config=PoolConfigSnapshot(
                pool_size=4,
                max_capacity_retry_seconds=30.0,
                dispatch_delay_at_completion_ms=10.0,
            ),
            pool_stats=PoolStatsSnapshot(
                capacity_retries=1,
                successes=3,
                peak_delay_ms=20.0,
                current_delay_ms=10.0,
                total_throttle_time_ms=5.0,
                max_concurrent_reached=3,
            ),
            query_ordering=(
                QueryOrderEntry(submit_index=0, complete_index=1, buffer_wait_ms=2.0),
                QueryOrderEntry(submit_index=1, complete_index=0, buffer_wait_ms=0.0),
            ),
        )

        state = factory.execution.begin_node_state(
            token_id=token.token_id,
            node_id=node.node_id,
            run_id=run.run_id,
            step_index=0,
            input_data={},
        )
        factory.execution.complete_node_state(
            state_id=state.state_id,
            status=NodeStateStatus.COMPLETED,
            output_data={"result": "ok"},
            duration_ms=100.0,
            context_after=ctx,
        )

        fetched = factory.execution.get_node_state(state.state_id)
        assert fetched is not None
        assert isinstance(fetched, NodeStateCompleted)
        assert fetched.context_after_json is not None
        assert json.loads(fetched.context_after_json) == ctx.to_dict()
