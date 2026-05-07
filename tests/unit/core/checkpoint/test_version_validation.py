# tests/unit/core/checkpoint/test_version_validation.py
"""Unit tests for Bug #12: No checkpoint state version validation.

These tests verify that checkpoint state includes version information
and that resume fails gracefully with incompatible versions.

Pure logic tests — no DB, no I/O. Moved from integration/checkpoint/.
"""

from typing import Any
from unittest.mock import Mock

import pytest

from elspeth.contracts.aggregation_checkpoint import AggregationCheckpointState
from elspeth.contracts.errors import AuditIntegrityError
from elspeth.contracts.schema_contract import FieldContract, SchemaContract
from elspeth.contracts.types import NodeID
from elspeth.core.config import AggregationSettings, TriggerConfig
from elspeth.engine.executors import AggregationExecutor
from elspeth.engine.spans import SpanFactory


def _make_contract(data: dict[str, Any]) -> SchemaContract:
    """Create a contract from observed data fields."""
    fields = tuple(
        FieldContract(
            normalized_name=k,
            original_name=k,
            python_type=object,
            required=False,
            source="inferred",
        )
        for k in data
    )
    return SchemaContract(mode="OBSERVED", fields=fields, locked=True)


class TestCheckpointVersionValidation:
    """Integration tests for checkpoint version validation."""

    def test_checkpoint_state_includes_version(self) -> None:
        """Verify checkpoint state includes _version field.

        Scenario:
        1. Create AggregationExecutor
        2. Get checkpoint state
        3. Verify: State contains "_version" field with value "4.0"

        This is Bug #12 fix: checkpoint state must include version for
        future compatibility when checkpoint format changes.
        """
        span_factory = SpanFactory()
        executor = AggregationExecutor(
            execution=None,  # type: ignore
            span_factory=span_factory,
            step_resolver=lambda node_id: 1,
            run_id="test_run",
        )

        # Get checkpoint state — now returns typed DTO
        state = executor.get_checkpoint_state()
        assert isinstance(state, AggregationCheckpointState)

        # Verify version field
        assert state.version == "4.0", f"Expected version '4.0', got {state.version!r}"
        # Verify wire format includes _version key
        state_dict = state.to_dict()
        assert "_version" in state_dict, "Checkpoint wire format must include _version field (Bug #12 fix)"

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
            execution=None,  # type: ignore
            span_factory=span_factory,
            step_resolver=lambda node_id: 1,
            run_id="test_run",
        )

        # Attempt to restore with incompatible version (old version 1.1)
        # Parse via from_dict() to get a typed DTO with wrong version
        incompatible_state = AggregationCheckpointState.from_dict(
            {
                "_version": "1.1",  # Old version - rejected after PipelineRow migration
            }
        )

        with pytest.raises(AuditIntegrityError) as exc_info:
            executor.restore_from_checkpoint(incompatible_state)

        # Verify error message is clear
        error_msg = str(exc_info.value)
        assert "Incompatible checkpoint version" in error_msg
        assert "1.1" in error_msg
        assert "4.0" in error_msg
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
        # Attempt to parse checkpoint without version field (old format)
        # Validation now happens in from_dict(), not in restore_from_checkpoint()
        old_format_state: dict[str, Any] = {
            "test_node": {
                "tokens": [],
                "batch_id": None,
            },
        }

        with pytest.raises(AuditIntegrityError) as exc_info:
            AggregationCheckpointState.from_dict(old_format_state)

        # Verify error mentions corruption (missing _version is Tier 1 corruption)
        error_msg = str(exc_info.value)
        assert "Corrupted aggregation checkpoint" in error_msg
        assert "_version" in error_msg

    def test_restore_succeeds_with_valid_version(self) -> None:
        """Verify restore succeeds with matching checkpoint version.

        Scenario:
        1. Create AggregationExecutor with aggregation settings for test_node
        2. Create checkpoint with valid version
        3. Restore from checkpoint
        4. Verify: No errors, restoration succeeds

        This confirms valid checkpoints still work after Bug #12 fix.
        """
        span_factory = SpanFactory()
        # AggregationExecutor requires aggregation_settings for nodes that appear in checkpoint
        # The restore_from_checkpoint method expects _nodes entries to exist for each node
        node_id = NodeID("test_node")
        trigger = TriggerConfig(count=10)
        aggregation_settings = {
            node_id: AggregationSettings(name="test_agg", plugin="batch_stats", input="source_out", on_error="discard", trigger=trigger)
        }
        execution = Mock()
        execution.get_batch.return_value = Mock(aggregation_node_id=node_id)
        execution.get_batch_members.return_value = [Mock(token_id="tok-001", ordinal=0)]
        executor = AggregationExecutor(
            execution=execution,
            span_factory=span_factory,
            step_resolver=lambda node_id: 1,
            run_id="test_run",
            aggregation_settings=aggregation_settings,
        )

        # Valid checkpoint state with matching version — constructed via from_dict()
        contract = _make_contract({"value": 1})
        contract_version = contract.version_hash()
        valid_state = AggregationCheckpointState.from_dict(
            {
                "_version": "4.0",
                "test_node": {
                    "tokens": [
                        {
                            "token_id": "tok-001",
                            "row_id": "row-001",
                            "row_data": {"value": 1},
                            "branch_name": None,
                            "fork_group_id": None,
                            "join_group_id": None,
                            "expand_group_id": None,
                            "contract_version": contract_version,
                            "contract": contract.to_checkpoint_format(),
                        }
                    ],
                    "batch_id": "batch-001",
                    "elapsed_age_seconds": 0.0,
                    "count_fire_offset": None,
                    "condition_fire_offset": None,
                },
            }
        )

        # Should not raise any errors
        executor.restore_from_checkpoint(valid_state)

        # Verify executor state was updated
        assert executor.get_buffer_count(NodeID("test_node")) == 1  # Restored single buffered token

    def test_checkpoint_state_excludes_work_item_fields(self) -> None:
        """Checkpoint payload must never persist transient WorkItem traversal fields."""
        from elspeth.contracts import TokenInfo
        from elspeth.testing import make_row

        span_factory = SpanFactory()
        node_id = NodeID("test_node")
        aggregation_settings = {
            node_id: AggregationSettings(
                name="test_agg",
                plugin="batch_stats",
                input="source_out",
                on_error="discard",
                trigger=TriggerConfig(count=10),
            )
        }
        recorder = Mock()
        recorder.create_batch.return_value = Mock(batch_id="batch-001")
        recorder.add_batch_member.return_value = None
        executor = AggregationExecutor(
            execution=recorder,
            span_factory=span_factory,
            step_resolver=lambda node_id: 1,
            run_id="test_run",
            aggregation_settings=aggregation_settings,
        )

        token = TokenInfo(
            row_id="row-001",
            token_id="tok-001",
            row_data=make_row({"value": 1}, contract=_make_contract({"value": 1})),
        )
        executor.buffer_row(node_id, token)
        state = executor.get_checkpoint_state()

        # Access via typed DTO — verify no transient WorkItem fields leak into checkpoint
        node_ckpt = state.nodes["test_node"]
        token_dict = node_ckpt.tokens[0].to_dict()
        forbidden_fields = {
            "start_step",
            "current_node_id",
            "coalesce_at_step",
            "coalesce_node_id",
            "coalesce_name",
        }
        assert forbidden_fields.isdisjoint(token_dict.keys())
