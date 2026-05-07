from __future__ import annotations

import json

import pytest

from elspeth.contracts import (
    NodeStateCompleted,
    NodeStateFailed,
    NodeStateOpen,
    NodeStatePending,
    NodeStateStatus,
    NodeType,
    RoutingMode,
    RoutingSpec,
)
from elspeth.contracts.errors import ConfigGateReason
from elspeth.contracts.schema import SchemaConfig
from elspeth.core.landscape import LandscapeDB
from elspeth.core.landscape.factory import RecorderFactory
from tests.fixtures.landscape import make_recorder_with_run, register_test_node

_DYNAMIC_SCHEMA = SchemaConfig.from_dict({"mode": "observed"})


def _setup(*, run_id: str = "run-1") -> tuple[LandscapeDB, RecorderFactory]:
    setup = make_recorder_with_run(run_id=run_id, source_node_id="source-0")
    register_test_node(setup.data_flow, setup.run_id, "transform-1", node_type=NodeType.TRANSFORM, plugin_name="transform")
    return setup.db, setup.factory


def _setup_with_token(
    *,
    run_id: str = "run-1",
) -> tuple[LandscapeDB, RecorderFactory, str, str]:
    db, factory = _setup(run_id=run_id)
    row = factory.data_flow.create_row(run_id, "source-0", 0, {"name": "test"}, row_id="row-1")
    token = factory.data_flow.create_token("row-1", token_id="tok-1")
    return db, factory, row.row_id, token.token_id


def _setup_with_token_and_edge(
    *,
    run_id: str = "run-1",
) -> tuple[LandscapeDB, RecorderFactory, str, str, str]:
    db, factory, row_id, token_id = _setup_with_token(run_id=run_id)
    edge = factory.data_flow.register_edge(
        run_id,
        "source-0",
        "transform-1",
        "continue",
        RoutingMode.MOVE,
        edge_id="edge-1",
    )
    return db, factory, row_id, token_id, edge.edge_id


def _make_gate_reason(condition: str = "row['x'] > 0", result: str = "true") -> ConfigGateReason:
    return ConfigGateReason(condition=condition, result=result)


class TestBeginNodeState:
    def test_creates_open_state(self):
        _db, factory, _row_id, token_id = _setup_with_token()

        result = factory.execution.begin_node_state(
            token_id=token_id,
            node_id="source-0",
            run_id="run-1",
            step_index=0,
            input_data={"name": "test"},
        )

        assert isinstance(result, NodeStateOpen)
        assert result.status == NodeStateStatus.OPEN

    def test_returns_node_state_open_type(self):
        _db, factory, _row_id, token_id = _setup_with_token()

        result = factory.execution.begin_node_state(
            token_id=token_id,
            node_id="source-0",
            run_id="run-1",
            step_index=0,
            input_data={"x": 1},
        )

        assert type(result) is NodeStateOpen

    def test_generates_state_id(self):
        _db, factory, _row_id, token_id = _setup_with_token()

        result = factory.execution.begin_node_state(
            token_id=token_id,
            node_id="source-0",
            run_id="run-1",
            step_index=0,
            input_data={"x": 1},
        )

        assert result.state_id is not None
        assert isinstance(result.state_id, str)
        assert len(result.state_id) > 0

    def test_uses_explicit_state_id(self):
        _db, factory, _row_id, token_id = _setup_with_token()

        result = factory.execution.begin_node_state(
            token_id=token_id,
            node_id="source-0",
            run_id="run-1",
            step_index=0,
            input_data={"x": 1},
            state_id="my-state-id",
        )

        assert result.state_id == "my-state-id"

    def test_stores_input_hash(self):
        _db, factory, _row_id, token_id = _setup_with_token()

        result = factory.execution.begin_node_state(
            token_id=token_id,
            node_id="source-0",
            run_id="run-1",
            step_index=0,
            input_data={"name": "test"},
        )

        assert result.input_hash is not None
        assert isinstance(result.input_hash, str)
        assert len(result.input_hash) > 0

    def test_same_input_produces_same_hash(self):
        _db, factory, _row_id, token_id = _setup_with_token()

        result1 = factory.execution.begin_node_state(
            token_id=token_id,
            node_id="source-0",
            run_id="run-1",
            step_index=0,
            input_data={"a": 1, "b": 2},
            state_id="state-1",
        )
        # Use different node to avoid UNIQUE constraint on (token_id, node_id, attempt)
        result2 = factory.execution.begin_node_state(
            token_id=token_id,
            node_id="transform-1",
            run_id="run-1",
            step_index=1,
            input_data={"a": 1, "b": 2},
            state_id="state-2",
        )

        assert result1.input_hash == result2.input_hash

    def test_different_input_produces_different_hash(self):
        _db, factory, _row_id, token_id = _setup_with_token()

        result1 = factory.execution.begin_node_state(
            token_id=token_id,
            node_id="source-0",
            run_id="run-1",
            step_index=0,
            input_data={"a": 1},
            state_id="state-1",
        )
        # Use different node to avoid UNIQUE constraint on (token_id, node_id, attempt)
        result2 = factory.execution.begin_node_state(
            token_id=token_id,
            node_id="transform-1",
            run_id="run-1",
            step_index=1,
            input_data={"a": 2},
            state_id="state-2",
        )

        assert result1.input_hash != result2.input_hash

    def test_context_before_defaults_to_none(self):
        _db, factory, _row_id, token_id = _setup_with_token()

        result = factory.execution.begin_node_state(
            token_id=token_id,
            node_id="source-0",
            run_id="run-1",
            step_index=0,
            input_data={"x": 1},
        )

        fetched = factory.execution.get_node_state(result.state_id)
        assert fetched is not None
        assert fetched.context_before_json is None

    def test_records_token_id(self):
        _db, factory, _row_id, token_id = _setup_with_token()

        result = factory.execution.begin_node_state(
            token_id=token_id,
            node_id="source-0",
            run_id="run-1",
            step_index=0,
            input_data={"x": 1},
        )

        assert result.token_id == token_id

    def test_records_node_id(self):
        _db, factory, _row_id, token_id = _setup_with_token()

        result = factory.execution.begin_node_state(
            token_id=token_id,
            node_id="source-0",
            run_id="run-1",
            step_index=0,
            input_data={"x": 1},
        )

        assert result.node_id == "source-0"

    def test_run_id_persisted_in_database(self):
        _db, factory, _row_id, token_id = _setup_with_token()

        result = factory.execution.begin_node_state(
            token_id=token_id,
            node_id="source-0",
            run_id="run-1",
            step_index=0,
            input_data={"x": 1},
        )

        # run_id is not on the NodeStateOpen dataclass but is persisted in DB.
        # Verify via get_node_state returning a valid object (it was stored with run_id).
        fetched = factory.execution.get_node_state(result.state_id)
        assert fetched is not None
        assert fetched.state_id == result.state_id

    def test_records_step_index(self):
        _db, factory, _row_id, token_id = _setup_with_token()

        result = factory.execution.begin_node_state(
            token_id=token_id,
            node_id="source-0",
            run_id="run-1",
            step_index=3,
            input_data={"x": 1},
        )

        assert result.step_index == 3

    def test_records_attempt(self):
        _db, factory, _row_id, token_id = _setup_with_token()

        result = factory.execution.begin_node_state(
            token_id=token_id,
            node_id="source-0",
            run_id="run-1",
            step_index=0,
            input_data={"x": 1},
            attempt=2,
        )

        assert result.attempt == 2

    def test_attempt_defaults_to_zero(self):
        _db, factory, _row_id, token_id = _setup_with_token()

        result = factory.execution.begin_node_state(
            token_id=token_id,
            node_id="source-0",
            run_id="run-1",
            step_index=0,
            input_data={"x": 1},
        )

        assert result.attempt == 0

    def test_persists_to_database(self):
        _db, factory, _row_id, token_id = _setup_with_token()

        result = factory.execution.begin_node_state(
            token_id=token_id,
            node_id="source-0",
            run_id="run-1",
            step_index=0,
            input_data={"x": 1},
        )

        fetched = factory.execution.get_node_state(result.state_id)
        assert fetched is not None
        assert fetched.state_id == result.state_id
        assert fetched.status == NodeStateStatus.OPEN

    def test_multiple_states_for_same_token_different_nodes(self):
        _db, factory, _row_id, token_id = _setup_with_token()

        state1 = factory.execution.begin_node_state(
            token_id=token_id,
            node_id="source-0",
            run_id="run-1",
            step_index=0,
            input_data={"x": 1},
            state_id="state-1",
        )
        state2 = factory.execution.begin_node_state(
            token_id=token_id,
            node_id="transform-1",
            run_id="run-1",
            step_index=1,
            input_data={"x": 1},
            state_id="state-2",
        )

        assert state1.state_id != state2.state_id
        assert factory.execution.get_node_state(state1.state_id) is not None
        assert factory.execution.get_node_state(state2.state_id) is not None

    def test_started_at_is_set(self):
        _db, factory, _row_id, token_id = _setup_with_token()

        result = factory.execution.begin_node_state(
            token_id=token_id,
            node_id="source-0",
            run_id="run-1",
            step_index=0,
            input_data={"x": 1},
        )

        assert result.started_at is not None


class TestCompleteNodeState:
    def test_completes_to_completed_status(self):
        _db, factory, _row_id, token_id = _setup_with_token()

        state = factory.execution.begin_node_state(
            token_id=token_id,
            node_id="source-0",
            run_id="run-1",
            step_index=0,
            input_data={"x": 1},
        )
        result = factory.execution.complete_node_state(
            state.state_id,
            NodeStateStatus.COMPLETED,
            output_data={"x": 1, "y": 2},
            duration_ms=100,
        )

        assert isinstance(result, NodeStateCompleted)
        assert result.status == NodeStateStatus.COMPLETED

    def test_completes_to_failed_status(self):
        _db, factory, _row_id, token_id = _setup_with_token()

        state = factory.execution.begin_node_state(
            token_id=token_id,
            node_id="source-0",
            run_id="run-1",
            step_index=0,
            input_data={"x": 1},
        )
        result = factory.execution.complete_node_state(
            state.state_id,
            NodeStateStatus.FAILED,
            error={"reason": "something broke"},
            duration_ms=50,
        )

        assert isinstance(result, NodeStateFailed)
        assert result.status == NodeStateStatus.FAILED

    def test_completes_to_pending_status(self):
        _db, factory, _row_id, token_id = _setup_with_token()

        state = factory.execution.begin_node_state(
            token_id=token_id,
            node_id="source-0",
            run_id="run-1",
            step_index=0,
            input_data={"x": 1},
        )
        result = factory.execution.complete_node_state(
            state.state_id,
            NodeStateStatus.PENDING,
            duration_ms=25,
        )

        assert isinstance(result, NodeStatePending)
        assert result.status == NodeStateStatus.PENDING

    def test_raises_value_error_for_open_status(self):
        _db, factory, _row_id, token_id = _setup_with_token()

        state = factory.execution.begin_node_state(
            token_id=token_id,
            node_id="source-0",
            run_id="run-1",
            step_index=0,
            input_data={"x": 1},
        )

        with pytest.raises(ValueError):
            factory.execution.complete_node_state(
                state.state_id,
                NodeStateStatus.OPEN,
                duration_ms=10,
            )

    def test_raises_value_error_for_none_duration(self):
        _db, factory, _row_id, token_id = _setup_with_token()

        state = factory.execution.begin_node_state(
            token_id=token_id,
            node_id="source-0",
            run_id="run-1",
            step_index=0,
            input_data={"x": 1},
        )

        with pytest.raises(ValueError):
            factory.execution.complete_node_state(
                state.state_id,
                NodeStateStatus.COMPLETED,
                duration_ms=None,
            )

    def test_stores_output_hash(self):
        _db, factory, _row_id, token_id = _setup_with_token()

        state = factory.execution.begin_node_state(
            token_id=token_id,
            node_id="source-0",
            run_id="run-1",
            step_index=0,
            input_data={"x": 1},
        )
        factory.execution.complete_node_state(
            state.state_id,
            NodeStateStatus.COMPLETED,
            output_data={"x": 1, "result": "done"},
            duration_ms=100,
        )

        fetched = factory.execution.get_node_state(state.state_id)
        assert fetched is not None
        assert isinstance(fetched, NodeStateCompleted)
        assert fetched.output_hash is not None
        assert isinstance(fetched.output_hash, str)
        assert len(fetched.output_hash) > 0

    def test_same_output_produces_same_hash(self):
        _db, factory, _row_id, token_id = _setup_with_token()

        state1 = factory.execution.begin_node_state(
            token_id=token_id,
            node_id="source-0",
            run_id="run-1",
            step_index=0,
            input_data={"x": 1},
            state_id="state-1",
        )
        factory.execution.complete_node_state(
            state1.state_id,
            NodeStateStatus.COMPLETED,
            output_data={"result": "ok"},
            duration_ms=10,
        )

        state2 = factory.execution.begin_node_state(
            token_id=token_id,
            node_id="transform-1",
            run_id="run-1",
            step_index=1,
            input_data={"x": 1},
            state_id="state-2",
        )
        factory.execution.complete_node_state(
            state2.state_id,
            NodeStateStatus.COMPLETED,
            output_data={"result": "ok"},
            duration_ms=20,
        )

        fetched1 = factory.execution.get_node_state(state1.state_id)
        fetched2 = factory.execution.get_node_state(state2.state_id)
        assert isinstance(fetched1, NodeStateCompleted)
        assert isinstance(fetched2, NodeStateCompleted)
        assert fetched1.output_hash == fetched2.output_hash

    def test_stores_error_json(self):
        _db, factory, _row_id, token_id = _setup_with_token()

        state = factory.execution.begin_node_state(
            token_id=token_id,
            node_id="source-0",
            run_id="run-1",
            step_index=0,
            input_data={"x": 1},
        )
        error_data = {"reason": "timeout", "code": 504}
        factory.execution.complete_node_state(
            state.state_id,
            NodeStateStatus.FAILED,
            error=error_data,
            duration_ms=5000,
        )

        fetched = factory.execution.get_node_state(state.state_id)
        assert fetched is not None
        assert isinstance(fetched, NodeStateFailed)
        assert fetched.error_json is not None
        parsed = json.loads(fetched.error_json)
        assert parsed == error_data

    def test_stores_success_reason_json(self):
        _db, factory, _row_id, token_id = _setup_with_token()

        state = factory.execution.begin_node_state(
            token_id=token_id,
            node_id="source-0",
            run_id="run-1",
            step_index=0,
            input_data={"x": 1},
        )
        success = {"action": "classified", "confidence": 0.95}
        factory.execution.complete_node_state(
            state.state_id,
            NodeStateStatus.COMPLETED,
            output_data={"x": 1},
            duration_ms=200,
            success_reason=success,
        )

        fetched = factory.execution.get_node_state(state.state_id)
        assert fetched is not None
        assert isinstance(fetched, NodeStateCompleted)
        assert fetched.success_reason_json is not None
        parsed = json.loads(fetched.success_reason_json)
        assert parsed == success

    def test_stores_context_after_json(self):
        from elspeth.contracts.node_state_context import (
            PoolConfigSnapshot,
            PoolExecutionContext,
            PoolStatsSnapshot,
            QueryOrderEntry,
        )

        _db, factory, _row_id, token_id = _setup_with_token()

        state = factory.execution.begin_node_state(
            token_id=token_id,
            node_id="source-0",
            run_id="run-1",
            step_index=0,
            input_data={"x": 1},
        )
        context = PoolExecutionContext(
            pool_config=PoolConfigSnapshot(pool_size=4, max_capacity_retry_seconds=30.0, dispatch_delay_at_completion_ms=10.0),
            pool_stats=PoolStatsSnapshot(
                capacity_retries=0,
                successes=2,
                peak_delay_ms=15.0,
                current_delay_ms=10.0,
                total_throttle_time_ms=0.0,
                max_concurrent_reached=2,
            ),
            query_ordering=(QueryOrderEntry(submit_index=0, complete_index=0, buffer_wait_ms=0.0),),
        )
        factory.execution.complete_node_state(
            state.state_id,
            NodeStateStatus.COMPLETED,
            output_data={"x": 1},
            duration_ms=300,
            context_after=context,
        )

        fetched = factory.execution.get_node_state(state.state_id)
        assert fetched is not None
        assert isinstance(fetched, NodeStateCompleted)
        assert fetched.context_after_json is not None
        parsed = json.loads(fetched.context_after_json)
        assert parsed == context.to_dict()

    def test_stores_duration_ms(self):
        _db, factory, _row_id, token_id = _setup_with_token()

        state = factory.execution.begin_node_state(
            token_id=token_id,
            node_id="source-0",
            run_id="run-1",
            step_index=0,
            input_data={"x": 1},
        )
        factory.execution.complete_node_state(
            state.state_id,
            NodeStateStatus.COMPLETED,
            output_data={"x": 1},
            duration_ms=42,
        )

        fetched = factory.execution.get_node_state(state.state_id)
        assert fetched is not None
        assert isinstance(fetched, NodeStateCompleted)
        assert fetched.duration_ms == 42

    def test_completed_returns_correct_type(self):
        _db, factory, _row_id, token_id = _setup_with_token()

        state = factory.execution.begin_node_state(
            token_id=token_id,
            node_id="source-0",
            run_id="run-1",
            step_index=0,
            input_data={"x": 1},
        )
        result = factory.execution.complete_node_state(
            state.state_id,
            NodeStateStatus.COMPLETED,
            output_data={"x": 1},
            duration_ms=10,
        )

        assert type(result) is NodeStateCompleted

    def test_failed_returns_correct_type(self):
        _db, factory, _row_id, token_id = _setup_with_token()

        state = factory.execution.begin_node_state(
            token_id=token_id,
            node_id="source-0",
            run_id="run-1",
            step_index=0,
            input_data={"x": 1},
        )
        result = factory.execution.complete_node_state(
            state.state_id,
            NodeStateStatus.FAILED,
            error={"reason": "bad"},
            duration_ms=10,
        )

        assert type(result) is NodeStateFailed

    def test_pending_returns_correct_type(self):
        _db, factory, _row_id, token_id = _setup_with_token()

        state = factory.execution.begin_node_state(
            token_id=token_id,
            node_id="source-0",
            run_id="run-1",
            step_index=0,
            input_data={"x": 1},
        )
        result = factory.execution.complete_node_state(
            state.state_id,
            NodeStateStatus.PENDING,
            duration_ms=10,
        )

        assert type(result) is NodeStatePending

    def test_updates_status_in_database(self):
        _db, factory, _row_id, token_id = _setup_with_token()

        state = factory.execution.begin_node_state(
            token_id=token_id,
            node_id="source-0",
            run_id="run-1",
            step_index=0,
            input_data={"x": 1},
        )
        factory.execution.complete_node_state(
            state.state_id,
            NodeStateStatus.COMPLETED,
            output_data={"result": "ok"},
            duration_ms=100,
        )

        fetched = factory.execution.get_node_state(state.state_id)
        assert fetched is not None
        assert fetched.status == NodeStateStatus.COMPLETED

    def test_completed_has_completed_at(self):
        _db, factory, _row_id, token_id = _setup_with_token()

        state = factory.execution.begin_node_state(
            token_id=token_id,
            node_id="source-0",
            run_id="run-1",
            step_index=0,
            input_data={"x": 1},
        )
        factory.execution.complete_node_state(
            state.state_id,
            NodeStateStatus.COMPLETED,
            output_data={"x": 1},
            duration_ms=100,
        )

        fetched = factory.execution.get_node_state(state.state_id)
        assert isinstance(fetched, NodeStateCompleted)
        assert fetched.completed_at is not None

    def test_failed_has_completed_at(self):
        _db, factory, _row_id, token_id = _setup_with_token()

        state = factory.execution.begin_node_state(
            token_id=token_id,
            node_id="source-0",
            run_id="run-1",
            step_index=0,
            input_data={"x": 1},
        )
        factory.execution.complete_node_state(
            state.state_id,
            NodeStateStatus.FAILED,
            error={"reason": "bad"},
            duration_ms=50,
        )

        fetched = factory.execution.get_node_state(state.state_id)
        assert isinstance(fetched, NodeStateFailed)
        assert fetched.completed_at is not None

    def test_pending_has_completed_at(self):
        _db, factory, _row_id, token_id = _setup_with_token()

        state = factory.execution.begin_node_state(
            token_id=token_id,
            node_id="source-0",
            run_id="run-1",
            step_index=0,
            input_data={"x": 1},
        )
        factory.execution.complete_node_state(
            state.state_id,
            NodeStateStatus.PENDING,
            duration_ms=25,
        )

        fetched = factory.execution.get_node_state(state.state_id)
        assert isinstance(fetched, NodeStatePending)
        assert fetched.completed_at is not None

    def test_context_after_defaults_to_none(self):
        _db, factory, _row_id, token_id = _setup_with_token()

        state = factory.execution.begin_node_state(
            token_id=token_id,
            node_id="source-0",
            run_id="run-1",
            step_index=0,
            input_data={"x": 1},
        )
        factory.execution.complete_node_state(
            state.state_id,
            NodeStateStatus.COMPLETED,
            output_data={"x": 1},
            duration_ms=10,
        )

        fetched = factory.execution.get_node_state(state.state_id)
        assert isinstance(fetched, NodeStateCompleted)
        assert fetched.context_after_json is None

    def test_rejects_completed_without_output_data(self):
        _db, factory, _row_id, token_id = _setup_with_token()

        state = factory.execution.begin_node_state(
            token_id=token_id,
            node_id="source-0",
            run_id="run-1",
            step_index=0,
            input_data={"x": 1},
        )

        with pytest.raises(ValueError, match="COMPLETED node state requires output_data"):
            factory.execution.complete_node_state(
                state.state_id,
                NodeStateStatus.COMPLETED,
                output_data=None,
                duration_ms=10,
            )

    def test_rejects_failed_without_error(self):
        _db, factory, _row_id, token_id = _setup_with_token()

        state = factory.execution.begin_node_state(
            token_id=token_id,
            node_id="source-0",
            run_id="run-1",
            step_index=0,
            input_data={"x": 1},
        )

        with pytest.raises(ValueError, match="FAILED node state requires error"):
            factory.execution.complete_node_state(
                state.state_id,
                NodeStateStatus.FAILED,
                error=None,
                duration_ms=10,
            )

    def test_completed_with_valid_output_data_succeeds(self):
        _db, factory, _row_id, token_id = _setup_with_token()

        state = factory.execution.begin_node_state(
            token_id=token_id,
            node_id="source-0",
            run_id="run-1",
            step_index=0,
            input_data={"x": 1},
        )

        result = factory.execution.complete_node_state(
            state.state_id,
            NodeStateStatus.COMPLETED,
            output_data={"x": 1, "result": "ok"},
            duration_ms=10,
        )
        assert isinstance(result, NodeStateCompleted)

    def test_failed_with_valid_error_succeeds(self):
        _db, factory, _row_id, token_id = _setup_with_token()

        state = factory.execution.begin_node_state(
            token_id=token_id,
            node_id="source-0",
            run_id="run-1",
            step_index=0,
            input_data={"x": 1},
        )

        result = factory.execution.complete_node_state(
            state.state_id,
            NodeStateStatus.FAILED,
            error={"reason": "something broke"},
            duration_ms=10,
        )
        assert isinstance(result, NodeStateFailed)

    def test_error_json_defaults_to_none_for_failed(self):
        """FAILED without error is now rejected by pre-write validation.

        This test documents that the old behavior (FAILED with error=None
        silently producing error_json=None) is no longer allowed. See
        test_rejects_failed_without_error for the replacement test.
        """
        _db, factory, _row_id, token_id = _setup_with_token()

        state = factory.execution.begin_node_state(
            token_id=token_id,
            node_id="source-0",
            run_id="run-1",
            step_index=0,
            input_data={"x": 1},
        )

        with pytest.raises(ValueError, match="FAILED node state requires error"):
            factory.execution.complete_node_state(
                state.state_id,
                NodeStateStatus.FAILED,
                duration_ms=10,
            )

    def test_success_reason_defaults_to_none(self):
        _db, factory, _row_id, token_id = _setup_with_token()

        state = factory.execution.begin_node_state(
            token_id=token_id,
            node_id="source-0",
            run_id="run-1",
            step_index=0,
            input_data={"x": 1},
        )
        factory.execution.complete_node_state(
            state.state_id,
            NodeStateStatus.COMPLETED,
            output_data={"x": 1},
            duration_ms=10,
        )

        fetched = factory.execution.get_node_state(state.state_id)
        assert isinstance(fetched, NodeStateCompleted)
        assert fetched.success_reason_json is None


class TestGetNodeState:
    def test_returns_open_state(self):
        _db, factory, _row_id, token_id = _setup_with_token()

        state = factory.execution.begin_node_state(
            token_id=token_id,
            node_id="source-0",
            run_id="run-1",
            step_index=0,
            input_data={"x": 1},
        )

        fetched = factory.execution.get_node_state(state.state_id)
        assert fetched is not None
        assert isinstance(fetched, NodeStateOpen)
        assert fetched.status == NodeStateStatus.OPEN

    def test_returns_completed_state(self):
        _db, factory, _row_id, token_id = _setup_with_token()

        state = factory.execution.begin_node_state(
            token_id=token_id,
            node_id="source-0",
            run_id="run-1",
            step_index=0,
            input_data={"x": 1},
        )
        factory.execution.complete_node_state(
            state.state_id,
            NodeStateStatus.COMPLETED,
            output_data={"x": 1},
            duration_ms=50,
        )

        fetched = factory.execution.get_node_state(state.state_id)
        assert fetched is not None
        assert isinstance(fetched, NodeStateCompleted)
        assert fetched.status == NodeStateStatus.COMPLETED

    def test_returns_failed_state(self):
        _db, factory, _row_id, token_id = _setup_with_token()

        state = factory.execution.begin_node_state(
            token_id=token_id,
            node_id="source-0",
            run_id="run-1",
            step_index=0,
            input_data={"x": 1},
        )
        factory.execution.complete_node_state(
            state.state_id,
            NodeStateStatus.FAILED,
            error={"msg": "oops"},
            duration_ms=10,
        )

        fetched = factory.execution.get_node_state(state.state_id)
        assert fetched is not None
        assert isinstance(fetched, NodeStateFailed)
        assert fetched.status == NodeStateStatus.FAILED

    def test_returns_pending_state(self):
        _db, factory, _row_id, token_id = _setup_with_token()

        state = factory.execution.begin_node_state(
            token_id=token_id,
            node_id="source-0",
            run_id="run-1",
            step_index=0,
            input_data={"x": 1},
        )
        factory.execution.complete_node_state(
            state.state_id,
            NodeStateStatus.PENDING,
            duration_ms=5,
        )

        fetched = factory.execution.get_node_state(state.state_id)
        assert fetched is not None
        assert isinstance(fetched, NodeStatePending)
        assert fetched.status == NodeStateStatus.PENDING

    def test_returns_none_for_unknown_id(self):
        _db, factory = _setup()

        result = factory.execution.get_node_state("nonexistent-state-id")
        assert result is None

    def test_preserves_token_id(self):
        _db, factory, _row_id, token_id = _setup_with_token()

        state = factory.execution.begin_node_state(
            token_id=token_id,
            node_id="source-0",
            run_id="run-1",
            step_index=0,
            input_data={"x": 1},
        )

        fetched = factory.execution.get_node_state(state.state_id)
        assert fetched is not None
        assert fetched.token_id == token_id

    def test_preserves_node_id(self):
        _db, factory, _row_id, token_id = _setup_with_token()

        state = factory.execution.begin_node_state(
            token_id=token_id,
            node_id="transform-1",
            run_id="run-1",
            step_index=1,
            input_data={"x": 1},
        )

        fetched = factory.execution.get_node_state(state.state_id)
        assert fetched is not None
        assert fetched.node_id == "transform-1"

    def test_preserves_step_index(self):
        _db, factory, _row_id, token_id = _setup_with_token()

        state = factory.execution.begin_node_state(
            token_id=token_id,
            node_id="source-0",
            run_id="run-1",
            step_index=7,
            input_data={"x": 1},
        )

        fetched = factory.execution.get_node_state(state.state_id)
        assert fetched is not None
        assert fetched.step_index == 7

    def test_preserves_attempt(self):
        _db, factory, _row_id, token_id = _setup_with_token()

        state = factory.execution.begin_node_state(
            token_id=token_id,
            node_id="source-0",
            run_id="run-1",
            step_index=0,
            input_data={"x": 1},
            attempt=3,
        )

        fetched = factory.execution.get_node_state(state.state_id)
        assert fetched is not None
        assert fetched.attempt == 3

    def test_preserves_input_hash(self):
        _db, factory, _row_id, token_id = _setup_with_token()

        state = factory.execution.begin_node_state(
            token_id=token_id,
            node_id="source-0",
            run_id="run-1",
            step_index=0,
            input_data={"x": 1},
        )

        fetched = factory.execution.get_node_state(state.state_id)
        assert fetched is not None
        assert fetched.input_hash == state.input_hash

    def test_preserves_started_at(self):
        _db, factory, _row_id, token_id = _setup_with_token()

        state = factory.execution.begin_node_state(
            token_id=token_id,
            node_id="source-0",
            run_id="run-1",
            step_index=0,
            input_data={"x": 1},
        )

        fetched = factory.execution.get_node_state(state.state_id)
        assert fetched is not None
        # Compare without timezone info since SQLite may strip tzinfo on round-trip
        assert fetched.started_at.replace(tzinfo=None) == state.started_at.replace(tzinfo=None)


class TestRecordRoutingEvent:
    def test_records_event_with_mode_and_edge(self):
        _db, factory, _row_id, token_id, edge_id = _setup_with_token_and_edge()

        state = factory.execution.begin_node_state(
            token_id=token_id,
            node_id="source-0",
            run_id="run-1",
            step_index=0,
            input_data={"x": 1},
        )

        event = factory.execution.record_routing_event(
            state.state_id,
            edge_id,
            RoutingMode.MOVE,
        )

        assert event is not None
        assert event.edge_id == edge_id
        assert event.mode == RoutingMode.MOVE

    def test_generates_event_id(self):
        _db, factory, _row_id, token_id, edge_id = _setup_with_token_and_edge()

        state = factory.execution.begin_node_state(
            token_id=token_id,
            node_id="source-0",
            run_id="run-1",
            step_index=0,
            input_data={"x": 1},
        )

        event = factory.execution.record_routing_event(
            state.state_id,
            edge_id,
            RoutingMode.MOVE,
        )

        assert event.event_id is not None
        assert isinstance(event.event_id, str)
        assert len(event.event_id) > 0

    def test_uses_explicit_event_id(self):
        _db, factory, _row_id, token_id, edge_id = _setup_with_token_and_edge()

        state = factory.execution.begin_node_state(
            token_id=token_id,
            node_id="source-0",
            run_id="run-1",
            step_index=0,
            input_data={"x": 1},
        )

        event = factory.execution.record_routing_event(
            state.state_id,
            edge_id,
            RoutingMode.MOVE,
            event_id="my-event-id",
        )

        assert event.event_id == "my-event-id"

    def test_generates_routing_group_id(self):
        _db, factory, _row_id, token_id, edge_id = _setup_with_token_and_edge()

        state = factory.execution.begin_node_state(
            token_id=token_id,
            node_id="source-0",
            run_id="run-1",
            step_index=0,
            input_data={"x": 1},
        )

        event = factory.execution.record_routing_event(
            state.state_id,
            edge_id,
            RoutingMode.MOVE,
        )

        assert event.routing_group_id is not None
        assert isinstance(event.routing_group_id, str)
        assert len(event.routing_group_id) > 0

    def test_uses_explicit_routing_group_id(self):
        _db, factory, _row_id, token_id, edge_id = _setup_with_token_and_edge()

        state = factory.execution.begin_node_state(
            token_id=token_id,
            node_id="source-0",
            run_id="run-1",
            step_index=0,
            input_data={"x": 1},
        )

        event = factory.execution.record_routing_event(
            state.state_id,
            edge_id,
            RoutingMode.MOVE,
            routing_group_id="group-1",
        )

        assert event.routing_group_id == "group-1"

    def test_records_reason_as_hash(self):
        _db, factory, _row_id, token_id, edge_id = _setup_with_token_and_edge()

        state = factory.execution.begin_node_state(
            token_id=token_id,
            node_id="source-0",
            run_id="run-1",
            step_index=0,
            input_data={"x": 1},
        )

        reason = _make_gate_reason()
        event = factory.execution.record_routing_event(
            state.state_id,
            edge_id,
            RoutingMode.MOVE,
            reason=reason,
        )

        assert event.reason_hash is not None
        assert isinstance(event.reason_hash, str)
        assert len(event.reason_hash) > 0

    def test_reason_hash_none_when_no_reason(self):
        _db, factory, _row_id, token_id, edge_id = _setup_with_token_and_edge()

        state = factory.execution.begin_node_state(
            token_id=token_id,
            node_id="source-0",
            run_id="run-1",
            step_index=0,
            input_data={"x": 1},
        )

        event = factory.execution.record_routing_event(
            state.state_id,
            edge_id,
            RoutingMode.MOVE,
        )

        assert event.reason_hash is None

    def test_records_state_id(self):
        _db, factory, _row_id, token_id, edge_id = _setup_with_token_and_edge()

        state = factory.execution.begin_node_state(
            token_id=token_id,
            node_id="source-0",
            run_id="run-1",
            step_index=0,
            input_data={"x": 1},
        )

        event = factory.execution.record_routing_event(
            state.state_id,
            edge_id,
            RoutingMode.MOVE,
        )

        assert event.state_id == state.state_id

    def test_records_ordinal(self):
        _db, factory, _row_id, token_id, edge_id = _setup_with_token_and_edge()

        state = factory.execution.begin_node_state(
            token_id=token_id,
            node_id="source-0",
            run_id="run-1",
            step_index=0,
            input_data={"x": 1},
        )

        event = factory.execution.record_routing_event(
            state.state_id,
            edge_id,
            RoutingMode.MOVE,
            ordinal=5,
        )

        assert event.ordinal == 5

    def test_ordinal_defaults_to_zero(self):
        _db, factory, _row_id, token_id, edge_id = _setup_with_token_and_edge()

        state = factory.execution.begin_node_state(
            token_id=token_id,
            node_id="source-0",
            run_id="run-1",
            step_index=0,
            input_data={"x": 1},
        )

        event = factory.execution.record_routing_event(
            state.state_id,
            edge_id,
            RoutingMode.MOVE,
        )

        assert event.ordinal == 0

    def test_records_created_at(self):
        _db, factory, _row_id, token_id, edge_id = _setup_with_token_and_edge()

        state = factory.execution.begin_node_state(
            token_id=token_id,
            node_id="source-0",
            run_id="run-1",
            step_index=0,
            input_data={"x": 1},
        )

        event = factory.execution.record_routing_event(
            state.state_id,
            edge_id,
            RoutingMode.MOVE,
        )

        assert event.created_at is not None

    def test_copy_mode(self):
        _db, factory, _row_id, token_id, edge_id = _setup_with_token_and_edge()

        state = factory.execution.begin_node_state(
            token_id=token_id,
            node_id="source-0",
            run_id="run-1",
            step_index=0,
            input_data={"x": 1},
        )

        event = factory.execution.record_routing_event(
            state.state_id,
            edge_id,
            RoutingMode.COPY,
        )

        assert event.mode == RoutingMode.COPY


class TestRecordRoutingEvents:
    def test_records_multiple_events(self):
        _db, factory, _row_id, token_id, _edge_id = _setup_with_token_and_edge()

        # Register a second edge for the second route
        factory.data_flow.register_node(
            run_id="run-1",
            plugin_name="sink",
            node_type=NodeType.SINK,
            plugin_version="1.0",
            config={},
            node_id="sink-2",
            schema_config=_DYNAMIC_SCHEMA,
        )
        factory.data_flow.register_edge(
            "run-1",
            "source-0",
            "sink-2",
            "route_a",
            RoutingMode.COPY,
            edge_id="edge-2",
        )

        state = factory.execution.begin_node_state(
            token_id=token_id,
            node_id="source-0",
            run_id="run-1",
            step_index=0,
            input_data={"x": 1},
        )

        routes = [
            RoutingSpec(edge_id="edge-1", mode=RoutingMode.MOVE),
            RoutingSpec(edge_id="edge-2", mode=RoutingMode.COPY),
        ]
        events = factory.execution.record_routing_events(state.state_id, routes)

        assert len(events) == 2

    def test_events_share_routing_group_id(self):
        _db, factory, _row_id, token_id, _edge_id = _setup_with_token_and_edge()

        factory.data_flow.register_node(
            run_id="run-1",
            plugin_name="sink",
            node_type=NodeType.SINK,
            plugin_version="1.0",
            config={},
            node_id="sink-2",
            schema_config=_DYNAMIC_SCHEMA,
        )
        factory.data_flow.register_edge(
            "run-1",
            "source-0",
            "sink-2",
            "route_a",
            RoutingMode.COPY,
            edge_id="edge-2",
        )

        state = factory.execution.begin_node_state(
            token_id=token_id,
            node_id="source-0",
            run_id="run-1",
            step_index=0,
            input_data={"x": 1},
        )

        routes = [
            RoutingSpec(edge_id="edge-1", mode=RoutingMode.MOVE),
            RoutingSpec(edge_id="edge-2", mode=RoutingMode.COPY),
        ]
        events = factory.execution.record_routing_events(state.state_id, routes)

        group_ids = {e.routing_group_id for e in events}
        assert len(group_ids) == 1
        assert None not in group_ids

    def test_ordinals_are_sequential(self):
        _db, factory, _row_id, token_id, _edge_id = _setup_with_token_and_edge()

        factory.data_flow.register_node(
            run_id="run-1",
            plugin_name="sink",
            node_type=NodeType.SINK,
            plugin_version="1.0",
            config={},
            node_id="sink-2",
            schema_config=_DYNAMIC_SCHEMA,
        )
        factory.data_flow.register_edge(
            "run-1",
            "source-0",
            "sink-2",
            "route_a",
            RoutingMode.COPY,
            edge_id="edge-2",
        )

        factory.data_flow.register_node(
            run_id="run-1",
            plugin_name="sink3",
            node_type=NodeType.SINK,
            plugin_version="1.0",
            config={},
            node_id="sink-3",
            schema_config=_DYNAMIC_SCHEMA,
        )
        factory.data_flow.register_edge(
            "run-1",
            "source-0",
            "sink-3",
            "route_b",
            RoutingMode.COPY,
            edge_id="edge-3",
        )

        state = factory.execution.begin_node_state(
            token_id=token_id,
            node_id="source-0",
            run_id="run-1",
            step_index=0,
            input_data={"x": 1},
        )

        routes = [
            RoutingSpec(edge_id="edge-1", mode=RoutingMode.MOVE),
            RoutingSpec(edge_id="edge-2", mode=RoutingMode.COPY),
            RoutingSpec(edge_id="edge-3", mode=RoutingMode.COPY),
        ]
        events = factory.execution.record_routing_events(state.state_id, routes)

        ordinals = [e.ordinal for e in events]
        assert ordinals == [0, 1, 2]

    def test_events_have_correct_edge_ids(self):
        _db, factory, _row_id, token_id, _edge_id = _setup_with_token_and_edge()

        factory.data_flow.register_node(
            run_id="run-1",
            plugin_name="sink",
            node_type=NodeType.SINK,
            plugin_version="1.0",
            config={},
            node_id="sink-2",
            schema_config=_DYNAMIC_SCHEMA,
        )
        factory.data_flow.register_edge(
            "run-1",
            "source-0",
            "sink-2",
            "route_a",
            RoutingMode.COPY,
            edge_id="edge-2",
        )

        state = factory.execution.begin_node_state(
            token_id=token_id,
            node_id="source-0",
            run_id="run-1",
            step_index=0,
            input_data={"x": 1},
        )

        routes = [
            RoutingSpec(edge_id="edge-1", mode=RoutingMode.MOVE),
            RoutingSpec(edge_id="edge-2", mode=RoutingMode.COPY),
        ]
        events = factory.execution.record_routing_events(state.state_id, routes)

        edge_ids = [e.edge_id for e in events]
        assert edge_ids == ["edge-1", "edge-2"]

    def test_events_have_correct_modes(self):
        _db, factory, _row_id, token_id, _edge_id = _setup_with_token_and_edge()

        factory.data_flow.register_node(
            run_id="run-1",
            plugin_name="sink",
            node_type=NodeType.SINK,
            plugin_version="1.0",
            config={},
            node_id="sink-2",
            schema_config=_DYNAMIC_SCHEMA,
        )
        factory.data_flow.register_edge(
            "run-1",
            "source-0",
            "sink-2",
            "route_a",
            RoutingMode.COPY,
            edge_id="edge-2",
        )

        state = factory.execution.begin_node_state(
            token_id=token_id,
            node_id="source-0",
            run_id="run-1",
            step_index=0,
            input_data={"x": 1},
        )

        routes = [
            RoutingSpec(edge_id="edge-1", mode=RoutingMode.MOVE),
            RoutingSpec(edge_id="edge-2", mode=RoutingMode.COPY),
        ]
        events = factory.execution.record_routing_events(state.state_id, routes)

        modes = [e.mode for e in events]
        assert modes == [RoutingMode.MOVE, RoutingMode.COPY]

    def test_records_reason_hash_on_all_events(self):
        _db, factory, _row_id, token_id, _edge_id = _setup_with_token_and_edge()

        factory.data_flow.register_node(
            run_id="run-1",
            plugin_name="sink",
            node_type=NodeType.SINK,
            plugin_version="1.0",
            config={},
            node_id="sink-2",
            schema_config=_DYNAMIC_SCHEMA,
        )
        factory.data_flow.register_edge(
            "run-1",
            "source-0",
            "sink-2",
            "route_a",
            RoutingMode.COPY,
            edge_id="edge-2",
        )

        state = factory.execution.begin_node_state(
            token_id=token_id,
            node_id="source-0",
            run_id="run-1",
            step_index=0,
            input_data={"x": 1},
        )

        reason = _make_gate_reason(condition="row['x'] > 0", result="true")
        routes = [
            RoutingSpec(edge_id="edge-1", mode=RoutingMode.MOVE),
            RoutingSpec(edge_id="edge-2", mode=RoutingMode.COPY),
        ]
        events = factory.execution.record_routing_events(
            state.state_id,
            routes,
            reason=reason,
        )

        for event in events:
            assert event.reason_hash is not None
            assert isinstance(event.reason_hash, str)

    def test_all_events_share_same_reason_hash(self):
        _db, factory, _row_id, token_id, _edge_id = _setup_with_token_and_edge()

        factory.data_flow.register_node(
            run_id="run-1",
            plugin_name="sink",
            node_type=NodeType.SINK,
            plugin_version="1.0",
            config={},
            node_id="sink-2",
            schema_config=_DYNAMIC_SCHEMA,
        )
        factory.data_flow.register_edge(
            "run-1",
            "source-0",
            "sink-2",
            "route_a",
            RoutingMode.COPY,
            edge_id="edge-2",
        )

        state = factory.execution.begin_node_state(
            token_id=token_id,
            node_id="source-0",
            run_id="run-1",
            step_index=0,
            input_data={"x": 1},
        )

        reason = _make_gate_reason()
        routes = [
            RoutingSpec(edge_id="edge-1", mode=RoutingMode.MOVE),
            RoutingSpec(edge_id="edge-2", mode=RoutingMode.COPY),
        ]
        events = factory.execution.record_routing_events(
            state.state_id,
            routes,
            reason=reason,
        )

        hashes = {e.reason_hash for e in events}
        assert len(hashes) == 1

    def test_reason_hash_none_when_no_reason(self):
        _db, factory, _row_id, token_id, _edge_id = _setup_with_token_and_edge()

        factory.data_flow.register_node(
            run_id="run-1",
            plugin_name="sink",
            node_type=NodeType.SINK,
            plugin_version="1.0",
            config={},
            node_id="sink-2",
            schema_config=_DYNAMIC_SCHEMA,
        )
        factory.data_flow.register_edge(
            "run-1",
            "source-0",
            "sink-2",
            "route_a",
            RoutingMode.COPY,
            edge_id="edge-2",
        )

        state = factory.execution.begin_node_state(
            token_id=token_id,
            node_id="source-0",
            run_id="run-1",
            step_index=0,
            input_data={"x": 1},
        )

        routes = [
            RoutingSpec(edge_id="edge-1", mode=RoutingMode.MOVE),
            RoutingSpec(edge_id="edge-2", mode=RoutingMode.COPY),
        ]
        events = factory.execution.record_routing_events(state.state_id, routes)

        for event in events:
            assert event.reason_hash is None

    def test_empty_routes_returns_empty_list(self):
        _db, factory, _row_id, token_id, _edge_id = _setup_with_token_and_edge()

        state = factory.execution.begin_node_state(
            token_id=token_id,
            node_id="source-0",
            run_id="run-1",
            step_index=0,
            input_data={"x": 1},
        )

        events = factory.execution.record_routing_events(state.state_id, [])

        assert events == []

    def test_single_route_works(self):
        _db, factory, _row_id, token_id, _edge_id = _setup_with_token_and_edge()

        state = factory.execution.begin_node_state(
            token_id=token_id,
            node_id="source-0",
            run_id="run-1",
            step_index=0,
            input_data={"x": 1},
        )

        routes = [RoutingSpec(edge_id="edge-1", mode=RoutingMode.MOVE)]
        events = factory.execution.record_routing_events(state.state_id, routes)

        assert len(events) == 1
        assert events[0].edge_id == "edge-1"
        assert events[0].mode == RoutingMode.MOVE
        assert events[0].ordinal == 0

    def test_events_have_distinct_event_ids(self):
        _db, factory, _row_id, token_id, _edge_id = _setup_with_token_and_edge()

        factory.data_flow.register_node(
            run_id="run-1",
            plugin_name="sink",
            node_type=NodeType.SINK,
            plugin_version="1.0",
            config={},
            node_id="sink-2",
            schema_config=_DYNAMIC_SCHEMA,
        )
        factory.data_flow.register_edge(
            "run-1",
            "source-0",
            "sink-2",
            "route_a",
            RoutingMode.COPY,
            edge_id="edge-2",
        )

        state = factory.execution.begin_node_state(
            token_id=token_id,
            node_id="source-0",
            run_id="run-1",
            step_index=0,
            input_data={"x": 1},
        )

        routes = [
            RoutingSpec(edge_id="edge-1", mode=RoutingMode.MOVE),
            RoutingSpec(edge_id="edge-2", mode=RoutingMode.COPY),
        ]
        events = factory.execution.record_routing_events(state.state_id, routes)

        event_ids = [e.event_id for e in events]
        assert len(set(event_ids)) == 2

    def test_all_events_reference_same_state(self):
        _db, factory, _row_id, token_id, _edge_id = _setup_with_token_and_edge()

        factory.data_flow.register_node(
            run_id="run-1",
            plugin_name="sink",
            node_type=NodeType.SINK,
            plugin_version="1.0",
            config={},
            node_id="sink-2",
            schema_config=_DYNAMIC_SCHEMA,
        )
        factory.data_flow.register_edge(
            "run-1",
            "source-0",
            "sink-2",
            "route_a",
            RoutingMode.COPY,
            edge_id="edge-2",
        )

        state = factory.execution.begin_node_state(
            token_id=token_id,
            node_id="source-0",
            run_id="run-1",
            step_index=0,
            input_data={"x": 1},
        )

        routes = [
            RoutingSpec(edge_id="edge-1", mode=RoutingMode.MOVE),
            RoutingSpec(edge_id="edge-2", mode=RoutingMode.COPY),
        ]
        events = factory.execution.record_routing_events(state.state_id, routes)

        for event in events:
            assert event.state_id == state.state_id

    def test_all_events_have_created_at(self):
        _db, factory, _row_id, token_id, _edge_id = _setup_with_token_and_edge()

        factory.data_flow.register_node(
            run_id="run-1",
            plugin_name="sink",
            node_type=NodeType.SINK,
            plugin_version="1.0",
            config={},
            node_id="sink-2",
            schema_config=_DYNAMIC_SCHEMA,
        )
        factory.data_flow.register_edge(
            "run-1",
            "source-0",
            "sink-2",
            "route_a",
            RoutingMode.COPY,
            edge_id="edge-2",
        )

        state = factory.execution.begin_node_state(
            token_id=token_id,
            node_id="source-0",
            run_id="run-1",
            step_index=0,
            input_data={"x": 1},
        )

        routes = [
            RoutingSpec(edge_id="edge-1", mode=RoutingMode.MOVE),
            RoutingSpec(edge_id="edge-2", mode=RoutingMode.COPY),
        ]
        events = factory.execution.record_routing_events(state.state_id, routes)

        for event in events:
            assert event.created_at is not None

    def test_empty_routes_returns_empty_without_orphaned_payload(self, tmp_path):
        """Bug ut1w: empty routes must return early without storing payload."""
        from unittest.mock import MagicMock

        _db, factory, _row_id, token_id, _edge_id = _setup_with_token_and_edge()

        state = factory.execution.begin_node_state(
            token_id=token_id,
            node_id="source-0",
            run_id="run-1",
            step_index=0,
            input_data={"x": 1},
        )

        # Mock the payload store to detect any store() calls
        mock_store = MagicMock()
        factory.execution._payload_store = mock_store

        reason = {"action": "continue", "match": "default"}
        events = factory.execution.record_routing_events(state.state_id, routes=[], reason=reason)

        assert events == []
        mock_store.store.assert_not_called()
