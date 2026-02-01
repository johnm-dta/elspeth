# tests/integration/test_checkpoint_version_validation.py
"""Integration tests for Bug #12: No checkpoint state version validation.

These tests verify that checkpoint state includes version information
and that resume fails gracefully with incompatible versions.
"""

from typing import Any

import pytest

from elspeth.contracts.types import NodeID
from elspeth.engine.executors import AggregationExecutor
from elspeth.engine.spans import SpanFactory


class TestCheckpointVersionValidation:
    """Integration tests for checkpoint version validation."""

    def test_checkpoint_state_includes_version(self) -> None:
        """Verify checkpoint state includes _version field.

        Scenario:
        1. Create AggregationExecutor
        2. Get checkpoint state
        3. Verify: State contains "_version" field with value "1.1"

        This is Bug #12 fix: checkpoint state must include version for
        future compatibility when checkpoint format changes.
        """
        span_factory = SpanFactory()
        executor = AggregationExecutor(
            recorder=None,  # type: ignore
            span_factory=span_factory,
            run_id="test_run",
        )

        # Get checkpoint state
        state = executor.get_checkpoint_state()

        # Verify version field exists
        assert "_version" in state, "Checkpoint state must include _version field (Bug #12 fix)"
        assert state["_version"] == "1.1", f"Expected version '1.1', got {state['_version']!r}"

    def test_restore_requires_matching_version(self) -> None:
        """Verify restore fails with incompatible checkpoint version.

        Scenario:
        1. Create AggregationExecutor
        2. Attempt to restore from checkpoint with wrong version
        3. Verify: ValueError raised with clear error message

        This is Bug #12 fix: resume must validate checkpoint version
        to prevent cryptic errors when format changes.
        """
        span_factory = SpanFactory()
        executor = AggregationExecutor(
            recorder=None,  # type: ignore
            span_factory=span_factory,
            run_id="test_run",
        )

        # Attempt to restore with incompatible version
        incompatible_state = {
            "_version": "2.0",  # Future version
            "test_node": {
                "tokens": [],
                "batch_id": None,
            },
        }

        with pytest.raises(ValueError) as exc_info:
            executor.restore_from_checkpoint(incompatible_state)

        # Verify error message is clear
        error_msg = str(exc_info.value)
        assert "Incompatible checkpoint version" in error_msg
        assert "2.0" in error_msg
        assert "1.1" in error_msg
        assert "Cannot resume" in error_msg

    def test_restore_fails_without_version(self) -> None:
        """Verify restore fails if checkpoint lacks version field.

        Scenario:
        1. Create AggregationExecutor
        2. Attempt to restore from checkpoint without _version field
        3. Verify: ValueError raised (old checkpoint format)

        This ensures old checkpoints (before Bug #12 fix) are rejected
        rather than silently causing issues.
        """
        span_factory = SpanFactory()
        executor = AggregationExecutor(
            recorder=None,  # type: ignore
            span_factory=span_factory,
            run_id="test_run",
        )

        # Attempt to restore without version field (old format)
        old_format_state: dict[str, Any] = {
            "test_node": {
                "tokens": [],
                "batch_id": None,
            },
        }

        with pytest.raises(ValueError) as exc_info:
            executor.restore_from_checkpoint(old_format_state)

        # Verify error mentions incompatible version (None != "1.0")
        error_msg = str(exc_info.value)
        assert "Incompatible checkpoint version" in error_msg

    def test_restore_succeeds_with_valid_version(self) -> None:
        """Verify restore succeeds with matching checkpoint version.

        Scenario:
        1. Create AggregationExecutor
        2. Create checkpoint with valid version
        3. Restore from checkpoint
        4. Verify: No errors, restoration succeeds

        This confirms valid checkpoints still work after Bug #12 fix.
        """
        span_factory = SpanFactory()
        executor = AggregationExecutor(
            recorder=None,  # type: ignore
            span_factory=span_factory,
            run_id="test_run",
        )

        # Valid checkpoint state with matching version
        valid_state = {
            "_version": "1.1",  # Matching version
            "test_node": {
                "tokens": [
                    {
                        "token_id": "tok-001",
                        "row_id": "row-001",
                        "row_data": {"value": 1},
                        "branch_name": None,
                    }
                ],
                "batch_id": "batch-001",
                "elapsed_age_seconds": 0.0,
                "count_fire_offset": None,  # P2-2026-02-01: Required in v1.1
                "condition_fire_offset": None,  # P2-2026-02-01: Required in v1.1
            },
        }

        # Should not raise any errors
        executor.restore_from_checkpoint(valid_state)

        # Verify executor state was updated
        assert executor.get_buffer_count(NodeID("test_node")) == 1  # Restored single buffered token
