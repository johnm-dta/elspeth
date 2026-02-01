"""Tests for LandscapeRecorder node state operations."""

from __future__ import annotations

from elspeth.contracts import NodeStateStatus, NodeType
from elspeth.contracts.schema import SchemaConfig

# Dynamic schema for tests that don't care about specific fields
DYNAMIC_SCHEMA = SchemaConfig.from_dict({"fields": "dynamic"})


class TestLandscapeRecorderNodeStates:
    """Node state recording (what happened at each node)."""

    def test_begin_node_state(self) -> None:
        from elspeth.contracts import NodeStateStatus
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.recorder import LandscapeRecorder

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")
        source = recorder.register_node(
            run_id=run.run_id,
            plugin_name="source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        row = recorder.create_row(
            run_id=run.run_id,
            source_node_id=source.node_id,
            row_index=0,
            data={},
        )
        token = recorder.create_token(row_id=row.row_id)

        state = recorder.begin_node_state(
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
        from elspeth.core.landscape.recorder import LandscapeRecorder

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")
        node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="transform",
            node_type=NodeType.TRANSFORM,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        row = recorder.create_row(
            run_id=run.run_id,
            source_node_id=node.node_id,
            row_index=0,
            data={},
        )
        token = recorder.create_token(row_id=row.row_id)

        input_data = {"x": 1, "y": 2}
        output_data = {"x": 1, "y": 2, "z": 3}

        state = recorder.begin_node_state(
            token_id=token.token_id,
            node_id=node.node_id,
            run_id=run.run_id,
            step_index=0,
            input_data=input_data,
        )

        # P1: Verify input_hash matches expected
        expected_input_hash = stable_hash(input_data)
        assert state.input_hash == expected_input_hash, f"input_hash mismatch: expected {expected_input_hash}, got {state.input_hash}"

        completed = recorder.complete_node_state(
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
        from elspeth.core.landscape.recorder import LandscapeRecorder

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")
        node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="transform",
            node_type=NodeType.TRANSFORM,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        row = recorder.create_row(
            run_id=run.run_id,
            source_node_id=node.node_id,
            row_index=0,
            data={},
        )
        token = recorder.create_token(row_id=row.row_id)
        state = recorder.begin_node_state(
            token_id=token.token_id,
            node_id=node.node_id,
            run_id=run.run_id,
            step_index=0,
            input_data={"x": 1},
        )

        completed = recorder.complete_node_state(
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
        from elspeth.core.landscape.recorder import LandscapeRecorder

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")
        node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="transform",
            node_type=NodeType.TRANSFORM,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        row = recorder.create_row(
            run_id=run.run_id,
            source_node_id=node.node_id,
            row_index=0,
            data={},
        )
        token = recorder.create_token(row_id=row.row_id)
        state = recorder.begin_node_state(
            token_id=token.token_id,
            node_id=node.node_id,
            run_id=run.run_id,
            step_index=0,
            input_data={},
        )

        completed = recorder.complete_node_state(
            state_id=state.state_id,
            status=NodeStateStatus.FAILED,
            error={"message": "Validation failed", "code": "E001"},
            duration_ms=5.0,
        )

        assert completed.status == NodeStateStatus.FAILED
        assert completed.error_json is not None
        assert "Validation failed" in completed.error_json

    def test_complete_node_state_with_empty_output(self) -> None:
        """Empty dict output is valid and must produce non-NULL output_hash.

        Bug: P1-2026-01-19-complete-node-state-empty-output-hash
        """
        from elspeth.contracts import NodeStateStatus
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.recorder import LandscapeRecorder

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")
        node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="transform",
            node_type=NodeType.TRANSFORM,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        row = recorder.create_row(
            run_id=run.run_id,
            source_node_id=node.node_id,
            row_index=0,
            data={},
        )
        token = recorder.create_token(row_id=row.row_id)
        state = recorder.begin_node_state(
            token_id=token.token_id,
            node_id=node.node_id,
            run_id=run.run_id,
            step_index=0,
            input_data={},
        )

        # Empty output_data={} should succeed, not crash
        completed = recorder.complete_node_state(
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
        from elspeth.core.landscape.recorder import LandscapeRecorder

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")
        node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="transform",
            node_type=NodeType.TRANSFORM,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        row = recorder.create_row(
            run_id=run.run_id,
            source_node_id=node.node_id,
            row_index=0,
            data={},
        )
        token = recorder.create_token(row_id=row.row_id)
        state = recorder.begin_node_state(
            token_id=token.token_id,
            node_id=node.node_id,
            run_id=run.run_id,
            step_index=0,
            input_data={},
        )

        # Empty error={} should be serialized, not dropped
        completed = recorder.complete_node_state(
            state_id=state.state_id,
            status=NodeStateStatus.FAILED,
            error={},  # Empty dict error
            duration_ms=1.0,
        )

        assert completed.status == NodeStateStatus.FAILED
        assert completed.error_json == "{}"  # Empty dict serializes to "{}"

    def test_begin_node_state_with_empty_context(self) -> None:
        """Empty dict context_before is recorded, not dropped.

        Bug: P1-2026-01-19-complete-node-state-empty-output-hash
        """
        from elspeth.contracts import NodeStateStatus
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.recorder import LandscapeRecorder

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")
        node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="transform",
            node_type=NodeType.TRANSFORM,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        row = recorder.create_row(
            run_id=run.run_id,
            source_node_id=node.node_id,
            row_index=0,
            data={},
        )
        token = recorder.create_token(row_id=row.row_id)

        # Empty context_before={} should be serialized, not dropped
        state = recorder.begin_node_state(
            token_id=token.token_id,
            node_id=node.node_id,
            run_id=run.run_id,
            step_index=0,
            input_data={},
            context_before={},  # Empty dict context
        )

        assert state.status == NodeStateStatus.OPEN
        assert state.context_before_json == "{}"  # Empty dict serializes to "{}"

    def test_retry_increments_attempt(self) -> None:
        from elspeth.contracts import NodeStateStatus
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.recorder import LandscapeRecorder

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")
        node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="transform",
            node_type=NodeType.TRANSFORM,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        row = recorder.create_row(
            run_id=run.run_id,
            source_node_id=node.node_id,
            row_index=0,
            data={},
        )
        token = recorder.create_token(row_id=row.row_id)

        # First attempt fails
        state1 = recorder.begin_node_state(
            token_id=token.token_id,
            node_id=node.node_id,
            run_id=run.run_id,
            step_index=0,
            input_data={},
            attempt=0,
        )
        recorder.complete_node_state(state1.state_id, status=NodeStateStatus.FAILED, error={}, duration_ms=1.0)

        # Second attempt
        state2 = recorder.begin_node_state(
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
        from elspeth.core.landscape.recorder import LandscapeRecorder

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)

        # Create valid infrastructure
        run = recorder.begin_run(config={}, canonical_version="v1")
        node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="test_source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        row_record = recorder.create_row(
            run_id=run.run_id,
            source_node_id=node.node_id,
            row_index=0,
            data={"test": "value"},
        )
        token = recorder.create_token(row_id=row_record.row_id)

        # Create a completed state normally
        state = recorder.begin_node_state(
            token_id=token.token_id,
            node_id=node.node_id,
            run_id=run.run_id,
            step_index=0,
            input_data={"test": "data"},
        )
        recorder.complete_node_state(
            state_id=state.state_id,
            status=NodeStateStatus.COMPLETED,
            output_data={"result": "ok"},
            duration_ms=10.0,
        )

        # Verify it works normally
        retrieved = recorder.get_node_states_for_token(token.token_id)
        assert len(retrieved) == 1

        # Now corrupt the database - set completed_at to NULL
        with db.connection() as conn:
            conn.execute(
                text("UPDATE node_states SET completed_at = NULL WHERE state_id = :sid"),
                {"sid": state.state_id},
            )
            conn.commit()

        # Reading corrupted data should crash (Tier 1 rule)
        with pytest.raises(ValueError, match=r"NULL completed_at.*audit integrity violation"):
            recorder.get_node_states_for_token(token.token_id)

    def test_failed_state_with_null_completed_at_raises(self) -> None:
        """FAILED state with NULL completed_at raises integrity violation.

        Per Data Manifesto: "Bad data in audit trail = crash immediately"
        """
        import pytest
        from sqlalchemy import text

        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.recorder import LandscapeRecorder

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)

        # Create valid infrastructure
        run = recorder.begin_run(config={}, canonical_version="v1")
        node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="test_source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        row_record = recorder.create_row(
            run_id=run.run_id,
            source_node_id=node.node_id,
            row_index=0,
            data={"test": "value"},
        )
        token = recorder.create_token(row_id=row_record.row_id)

        # Create a failed state normally
        state = recorder.begin_node_state(
            token_id=token.token_id,
            node_id=node.node_id,
            run_id=run.run_id,
            step_index=0,
            input_data={"test": "data"},
        )
        recorder.complete_node_state(
            state_id=state.state_id,
            status=NodeStateStatus.FAILED,
            error={"message": "Something went wrong"},
            duration_ms=5.0,
        )

        # Verify it works normally
        retrieved = recorder.get_node_states_for_token(token.token_id)
        assert len(retrieved) == 1

        # Now corrupt the database - set completed_at to NULL
        with db.connection() as conn:
            conn.execute(
                text("UPDATE node_states SET completed_at = NULL WHERE state_id = :sid"),
                {"sid": state.state_id},
            )
            conn.commit()

        # Reading corrupted data should crash (Tier 1 rule)
        with pytest.raises(ValueError, match=r"NULL completed_at.*audit integrity violation"):
            recorder.get_node_states_for_token(token.token_id)


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
        from elspeth.core.landscape.recorder import LandscapeRecorder

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)

        run = recorder.begin_run(config={}, canonical_version="v1")
        node1 = recorder.register_node(
            run_id=run.run_id,
            plugin_name="test_transform_1",
            node_type=NodeType.TRANSFORM,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        node2 = recorder.register_node(
            run_id=run.run_id,
            plugin_name="test_transform_2",
            node_type=NodeType.TRANSFORM,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        row_record = recorder.create_row(
            run_id=run.run_id,
            source_node_id=node1.node_id,
            row_index=0,
            data={"test": "value"},
        )
        token = recorder.create_token(row_id=row_record.row_id)

        # Create states at step 0 with multiple attempts (simulating retries)
        # Insert OUT OF ORDER to test that ordering is enforced by the query
        state_0_attempt_1 = recorder.begin_node_state(
            token_id=token.token_id,
            node_id=node1.node_id,
            run_id=run.run_id,
            step_index=0,
            attempt=1,  # Second attempt first!
            input_data={"test": "data"},
        )
        recorder.complete_node_state(
            state_id=state_0_attempt_1.state_id,
            status=NodeStateStatus.FAILED,
            error={"message": "First failure"},
            duration_ms=10.0,
        )

        state_0_attempt_0 = recorder.begin_node_state(
            token_id=token.token_id,
            node_id=node1.node_id,
            run_id=run.run_id,
            step_index=0,
            attempt=0,  # First attempt second!
            input_data={"test": "data"},
        )
        recorder.complete_node_state(
            state_id=state_0_attempt_0.state_id,
            status=NodeStateStatus.COMPLETED,
            output_data={"result": "ok"},
            duration_ms=5.0,
        )

        # Create a state at step 1 using a different node (different step in pipeline)
        state_1_attempt_0 = recorder.begin_node_state(
            token_id=token.token_id,
            node_id=node2.node_id,  # Different node for step 1
            run_id=run.run_id,
            step_index=1,
            attempt=0,
            input_data={"test": "data2"},
        )
        recorder.complete_node_state(
            state_id=state_1_attempt_0.state_id,
            status=NodeStateStatus.COMPLETED,
            output_data={"result": "ok2"},
            duration_ms=3.0,
        )

        # REGRESSION CHECK: Verify ordering is (step_index, attempt)
        states = recorder.get_node_states_for_token(token.token_id)
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
