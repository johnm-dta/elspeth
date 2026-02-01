"""Tests for NodeStateRepository.

Tests the discriminated union handling for NodeState types:
- NodeStateOpen
- NodeStatePending
- NodeStateCompleted
- NodeStateFailed

Per Data Manifesto: Audit database is Tier 1 (FULL TRUST).
Invalid data must crash immediately - no coercion, no defaults.
"""

from dataclasses import dataclass
from datetime import UTC, datetime

import pytest

from elspeth.contracts.audit import (
    NodeStateCompleted,
    NodeStateFailed,
    NodeStateOpen,
    NodeStatePending,
)
from elspeth.contracts.enums import NodeStateStatus
from elspeth.core.landscape.repositories import NodeStateRepository


@dataclass
class NodeStateRow:
    """Mock SQLAlchemy row for node_states table.

    Simulates database rows that store status as string.
    """

    state_id: str
    token_id: str
    node_id: str
    step_index: int
    attempt: int
    status: str  # String in DB
    input_hash: str
    started_at: datetime
    context_before_json: str | None = None
    output_hash: str | None = None
    duration_ms: float | None = None
    completed_at: datetime | None = None
    context_after_json: str | None = None
    error_json: str | None = None
    success_reason_json: str | None = None  # TransformSuccessReason for successful transforms


class TestNodeStateRepositoryOpen:
    """Tests for NodeStateRepository loading OPEN states."""

    def test_load_open_state(self) -> None:
        """Load returns NodeStateOpen for OPEN status."""
        started_at = datetime.now(UTC)
        db_row = NodeStateRow(
            state_id="state_1",
            token_id="token_1",
            node_id="node_1",
            step_index=0,
            attempt=1,
            status="open",  # String from DB
            input_hash="hash_in",
            started_at=started_at,
            context_before_json='{"key": "value"}',
        )

        repo = NodeStateRepository(session=None)
        result = repo.load(db_row)

        assert isinstance(result, NodeStateOpen)
        assert result.state_id == "state_1"
        assert result.token_id == "token_1"
        assert result.node_id == "node_1"
        assert result.step_index == 0
        assert result.attempt == 1
        assert result.status == NodeStateStatus.OPEN
        assert result.input_hash == "hash_in"
        assert result.started_at == started_at
        assert result.context_before_json == '{"key": "value"}'

    def test_load_open_state_without_context(self) -> None:
        """Load handles OPEN state with NULL context_before_json."""
        db_row = NodeStateRow(
            state_id="state_1",
            token_id="token_1",
            node_id="node_1",
            step_index=0,
            attempt=1,
            status="open",
            input_hash="hash_in",
            started_at=datetime.now(UTC),
            context_before_json=None,  # NULL is valid for OPEN
        )

        repo = NodeStateRepository(session=None)
        result = repo.load(db_row)

        assert isinstance(result, NodeStateOpen)
        assert result.context_before_json is None


class TestNodeStateRepositoryPending:
    """Tests for NodeStateRepository loading PENDING states."""

    def test_load_pending_state(self) -> None:
        """Load returns NodeStatePending for PENDING status."""
        started_at = datetime.now(UTC)
        completed_at = datetime.now(UTC)
        db_row = NodeStateRow(
            state_id="state_1",
            token_id="token_1",
            node_id="node_1",
            step_index=1,
            attempt=1,
            status="pending",  # String from DB
            input_hash="hash_in",
            started_at=started_at,
            context_before_json='{"key": "value"}',
            duration_ms=100.5,
            completed_at=completed_at,
            context_after_json='{"async_ref": "batch_123"}',
        )

        repo = NodeStateRepository(session=None)
        result = repo.load(db_row)

        assert isinstance(result, NodeStatePending)
        assert result.state_id == "state_1"
        assert result.status == NodeStateStatus.PENDING
        assert result.duration_ms == 100.5
        assert result.completed_at == completed_at
        assert result.context_after_json == '{"async_ref": "batch_123"}'

    def test_load_crashes_on_pending_without_duration(self) -> None:
        """Load crashes if PENDING state has NULL duration_ms (Tier 1 violation)."""
        db_row = NodeStateRow(
            state_id="state_1",
            token_id="token_1",
            node_id="node_1",
            step_index=0,
            attempt=1,
            status="pending",
            input_hash="hash_in",
            started_at=datetime.now(UTC),
            completed_at=datetime.now(UTC),
            duration_ms=None,  # INVALID for PENDING
        )

        repo = NodeStateRepository(session=None)

        with pytest.raises(ValueError, match=r"PENDING.*duration_ms"):
            repo.load(db_row)

    def test_load_crashes_on_pending_without_completed_at(self) -> None:
        """Load crashes if PENDING state has NULL completed_at (Tier 1 violation)."""
        db_row = NodeStateRow(
            state_id="state_1",
            token_id="token_1",
            node_id="node_1",
            step_index=0,
            attempt=1,
            status="pending",
            input_hash="hash_in",
            started_at=datetime.now(UTC),
            duration_ms=100.0,
            completed_at=None,  # INVALID for PENDING
        )

        repo = NodeStateRepository(session=None)

        with pytest.raises(ValueError, match=r"PENDING.*completed_at"):
            repo.load(db_row)


class TestNodeStateRepositoryCompleted:
    """Tests for NodeStateRepository loading COMPLETED states."""

    def test_load_completed_state(self) -> None:
        """Load returns NodeStateCompleted for COMPLETED status."""
        started_at = datetime.now(UTC)
        completed_at = datetime.now(UTC)
        db_row = NodeStateRow(
            state_id="state_1",
            token_id="token_1",
            node_id="node_1",
            step_index=2,
            attempt=1,
            status="completed",  # String from DB
            input_hash="hash_in",
            started_at=started_at,
            context_before_json='{"key": "value"}',
            output_hash="hash_out",
            duration_ms=150.0,
            completed_at=completed_at,
            context_after_json='{"result": "ok"}',
        )

        repo = NodeStateRepository(session=None)
        result = repo.load(db_row)

        assert isinstance(result, NodeStateCompleted)
        assert result.state_id == "state_1"
        assert result.status == NodeStateStatus.COMPLETED
        assert result.output_hash == "hash_out"
        assert result.duration_ms == 150.0
        assert result.completed_at == completed_at
        assert result.context_after_json == '{"result": "ok"}'

    def test_load_crashes_on_completed_without_output_hash(self) -> None:
        """Load crashes if COMPLETED state has NULL output_hash (Tier 1 violation)."""
        db_row = NodeStateRow(
            state_id="state_1",
            token_id="token_1",
            node_id="node_1",
            step_index=0,
            attempt=1,
            status="completed",
            input_hash="hash_in",
            started_at=datetime.now(UTC),
            output_hash=None,  # INVALID for COMPLETED
            duration_ms=100.0,
            completed_at=datetime.now(UTC),
        )

        repo = NodeStateRepository(session=None)

        with pytest.raises(ValueError, match=r"COMPLETED.*output_hash"):
            repo.load(db_row)

    def test_load_crashes_on_completed_without_duration(self) -> None:
        """Load crashes if COMPLETED state has NULL duration_ms (Tier 1 violation)."""
        db_row = NodeStateRow(
            state_id="state_1",
            token_id="token_1",
            node_id="node_1",
            step_index=0,
            attempt=1,
            status="completed",
            input_hash="hash_in",
            started_at=datetime.now(UTC),
            output_hash="hash_out",
            duration_ms=None,  # INVALID for COMPLETED
            completed_at=datetime.now(UTC),
        )

        repo = NodeStateRepository(session=None)

        with pytest.raises(ValueError, match=r"COMPLETED.*duration_ms"):
            repo.load(db_row)

    def test_load_crashes_on_completed_without_completed_at(self) -> None:
        """Load crashes if COMPLETED state has NULL completed_at (Tier 1 violation)."""
        db_row = NodeStateRow(
            state_id="state_1",
            token_id="token_1",
            node_id="node_1",
            step_index=0,
            attempt=1,
            status="completed",
            input_hash="hash_in",
            started_at=datetime.now(UTC),
            output_hash="hash_out",
            duration_ms=100.0,
            completed_at=None,  # INVALID for COMPLETED
        )

        repo = NodeStateRepository(session=None)

        with pytest.raises(ValueError, match=r"COMPLETED.*completed_at"):
            repo.load(db_row)


class TestNodeStateRepositoryFailed:
    """Tests for NodeStateRepository loading FAILED states."""

    def test_load_failed_state(self) -> None:
        """Load returns NodeStateFailed for FAILED status."""
        started_at = datetime.now(UTC)
        completed_at = datetime.now(UTC)
        db_row = NodeStateRow(
            state_id="state_1",
            token_id="token_1",
            node_id="node_1",
            step_index=1,
            attempt=1,
            status="failed",  # String from DB
            input_hash="hash_in",
            started_at=started_at,
            context_before_json='{"key": "value"}',
            duration_ms=50.0,
            completed_at=completed_at,
            error_json='{"error": "boom", "code": 500}',
        )

        repo = NodeStateRepository(session=None)
        result = repo.load(db_row)

        assert isinstance(result, NodeStateFailed)
        assert result.state_id == "state_1"
        assert result.status == NodeStateStatus.FAILED
        assert result.duration_ms == 50.0
        assert result.completed_at == completed_at
        assert result.error_json == '{"error": "boom", "code": 500}'

    def test_load_failed_state_with_partial_output(self) -> None:
        """Load handles FAILED state with output_hash (partial results)."""
        db_row = NodeStateRow(
            state_id="state_1",
            token_id="token_1",
            node_id="node_1",
            step_index=0,
            attempt=1,
            status="failed",
            input_hash="hash_in",
            started_at=datetime.now(UTC),
            duration_ms=100.0,
            completed_at=datetime.now(UTC),
            output_hash="partial_hash",  # Partial results before failure
            context_after_json='{"partial": true}',
        )

        repo = NodeStateRepository(session=None)
        result = repo.load(db_row)

        assert isinstance(result, NodeStateFailed)
        assert result.output_hash == "partial_hash"
        assert result.context_after_json == '{"partial": true}'

    def test_load_crashes_on_failed_without_duration(self) -> None:
        """Load crashes if FAILED state has NULL duration_ms (Tier 1 violation)."""
        db_row = NodeStateRow(
            state_id="state_1",
            token_id="token_1",
            node_id="node_1",
            step_index=0,
            attempt=1,
            status="failed",
            input_hash="hash_in",
            started_at=datetime.now(UTC),
            duration_ms=None,  # INVALID for FAILED
            completed_at=datetime.now(UTC),
            error_json='{"error": "boom"}',
        )

        repo = NodeStateRepository(session=None)

        with pytest.raises(ValueError, match=r"FAILED.*duration_ms"):
            repo.load(db_row)

    def test_load_crashes_on_failed_without_completed_at(self) -> None:
        """Load crashes if FAILED state has NULL completed_at (Tier 1 violation)."""
        db_row = NodeStateRow(
            state_id="state_1",
            token_id="token_1",
            node_id="node_1",
            step_index=0,
            attempt=1,
            status="failed",
            input_hash="hash_in",
            started_at=datetime.now(UTC),
            duration_ms=100.0,
            completed_at=None,  # INVALID for FAILED
            error_json='{"error": "boom"}',
        )

        repo = NodeStateRepository(session=None)

        with pytest.raises(ValueError, match=r"FAILED.*completed_at"):
            repo.load(db_row)


class TestNodeStateRepositoryInvalidStatus:
    """Tests for NodeStateRepository handling invalid status values."""

    def test_load_crashes_on_invalid_status(self) -> None:
        """Load crashes on unknown status value per Data Manifesto."""
        db_row = NodeStateRow(
            state_id="state_1",
            token_id="token_1",
            node_id="node_1",
            step_index=0,
            attempt=1,
            status="invalid_garbage",  # Invalid!
            input_hash="hash_in",
            started_at=datetime.now(UTC),
        )

        repo = NodeStateRepository(session=None)

        with pytest.raises(ValueError):
            repo.load(db_row)

    def test_load_crashes_on_empty_status(self) -> None:
        """Load crashes on empty status string."""
        db_row = NodeStateRow(
            state_id="state_1",
            token_id="token_1",
            node_id="node_1",
            step_index=0,
            attempt=1,
            status="",  # Empty!
            input_hash="hash_in",
            started_at=datetime.now(UTC),
        )

        repo = NodeStateRepository(session=None)

        with pytest.raises(ValueError):
            repo.load(db_row)
