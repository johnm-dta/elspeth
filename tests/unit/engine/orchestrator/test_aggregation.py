# tests/unit/engine/orchestrator/test_aggregation.py
"""Tests for aggregation handling functions in the orchestrator.

aggregation.py handles:
- Finding batch-aware transforms by node ID
- Recovering incomplete batches after crash
- Checking and flushing aggregation timeouts (pre-row)
- Flushing remaining buffers at end-of-source

These are pure delegation functions — no internal state — tested via mocks.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import Any
from unittest.mock import Mock, create_autospec

import pytest

from elspeth.contracts import PendingOutcome, RowResult, TokenInfo
from elspeth.contracts.audit import Batch
from elspeth.contracts.enums import BatchStatus, TerminalOutcome, TerminalPath, TriggerType
from elspeth.contracts.errors import OrchestrationInvariantError
from elspeth.contracts.plugin_context import PluginContext
from elspeth.contracts.types import NodeID
from elspeth.core.landscape.execution_repository import ExecutionRepository
from elspeth.engine.orchestrator.aggregation import (
    check_aggregation_timeouts,
    find_aggregation_transform,
    flush_remaining_aggregation_buffers,
)
from elspeth.engine.orchestrator.ports import AggregationProcessorPort
from elspeth.engine.orchestrator.resume import handle_incomplete_batches
from elspeth.engine.orchestrator.run_state import AggNodeEntry
from elspeth.engine.orchestrator.types import PipelineConfig
from elspeth.engine.work_items import WorkItem
from elspeth.testing import make_row, make_token_info

# =============================================================================
# Helpers
# =============================================================================


@dataclass(frozen=True)
class _NamedPlugin:
    name: str


@dataclass(frozen=True)
class _AggregationSettings:
    name: str = "batch_agg"


@dataclass(frozen=True)
class _TransformPlaceholder:
    node_id: str


def _make_batch_transform(*, node_id: str, is_batch_aware: bool = True) -> Mock:
    """Create a mock transform satisfying TransformProtocol with batch awareness."""
    from elspeth.contracts import TransformProtocol

    transform = Mock(spec=TransformProtocol)
    transform.node_id = node_id
    transform.is_batch_aware = is_batch_aware
    transform.name = f"transform-{node_id}"
    return transform


def _make_config(
    *,
    transforms: list[Any] | None = None,
    aggregation_settings: dict[str, Any] | None = None,
) -> PipelineConfig:
    """Build a minimal PipelineConfig for aggregation tests."""
    return PipelineConfig(
        sources={"primary": _NamedPlugin(name="test-source")},
        transforms=transforms or [],
        sinks={"output": _NamedPlugin(name="output")},
        aggregation_settings=aggregation_settings or {},
    )


def _make_agg_settings(*, name: str = "batch_agg") -> _AggregationSettings:
    return _AggregationSettings(name=name)


def _make_result(
    outcome: TerminalOutcome | None,
    path: TerminalPath,
    *,
    token: TokenInfo | None = None,
    sink_name: str | None = None,
) -> RowResult:
    result_token = token or make_token_info()
    if path == TerminalPath.COALESCED and result_token.join_group_id is None:
        result_token = replace(result_token, join_group_id="join-1")
    return RowResult(
        token=result_token,
        final_data=make_row({}),
        outcome=outcome,
        path=path,
        sink_name=sink_name,
    )


def _make_work_item(
    *,
    token: TokenInfo | None = None,
    current_node_id: NodeID | None = None,
    coalesce_node_id: NodeID | None = None,
    coalesce_name: str | None = None,
) -> WorkItem:
    return WorkItem(
        token=token or make_token_info(),
        current_node_id=current_node_id if current_node_id is not None else NodeID("node-0"),
        coalesce_node_id=coalesce_node_id,
        coalesce_name=coalesce_name,
    )


def _make_pending() -> dict[str, list[tuple[TokenInfo, PendingOutcome | None]]]:
    return {"output": []}


def _make_batch(
    batch_id: str,
    status: BatchStatus,
    *,
    trigger_type: TriggerType | None = None,
    trigger_reason: str | None = None,
    aggregation_state_id: str | None = None,
) -> Batch:
    return Batch(
        batch_id=batch_id,
        run_id="run-1",
        aggregation_node_id="agg-1",
        attempt=0,
        status=status,
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        trigger_type=trigger_type,
        trigger_reason=trigger_reason,
        aggregation_state_id=aggregation_state_id,
    )


def _make_context() -> PluginContext:
    return PluginContext(run_id="run-1", config={})


def _make_processor() -> AggregationProcessorPort:
    return create_autospec(AggregationProcessorPort, instance=True, spec_set=True)


def _make_execution() -> ExecutionRepository:
    return create_autospec(ExecutionRepository, instance=True, spec_set=True)


# =============================================================================
# find_aggregation_transform
# =============================================================================


class TestFindAggregationTransform:
    """Tests for find_aggregation_transform()."""

    def test_finds_batch_aware_transform(self) -> None:
        """Returns transform and aggregation node ID for matching node_id."""
        t = _make_batch_transform(node_id="agg-node-1")
        config = _make_config(transforms=[_TransformPlaceholder("before"), t, _TransformPlaceholder("after")])

        result_transform, result_node_id = find_aggregation_transform(config, "agg-node-1", "batch1")

        assert result_transform is t
        assert result_node_id == NodeID("agg-node-1")

    def test_non_batch_aware_skipped(self) -> None:
        """Transform with is_batch_aware=False is not matched."""
        t = _make_batch_transform(node_id="agg-node-1", is_batch_aware=False)
        config = _make_config(transforms=[t])

        with pytest.raises(OrchestrationInvariantError, match="No batch-aware transform"):
            find_aggregation_transform(config, "agg-node-1", "batch1")

    def test_wrong_node_id_skipped(self) -> None:
        """Transform with different node_id is not matched."""
        t = _make_batch_transform(node_id="other-node")
        config = _make_config(transforms=[t])

        with pytest.raises(OrchestrationInvariantError, match="No batch-aware transform"):
            find_aggregation_transform(config, "agg-node-1", "batch1")

    def test_no_transforms_raises(self) -> None:
        """Empty transforms list raises with helpful error."""
        config = _make_config(transforms=[])

        with pytest.raises(OrchestrationInvariantError, match="No batch-aware transform"):
            find_aggregation_transform(config, "agg-node-1", "batch1")

    def test_error_includes_aggregation_name(self) -> None:
        """Error message includes the aggregation name for debugging."""
        config = _make_config(transforms=[])

        with pytest.raises(OrchestrationInvariantError, match="my_aggregation"):
            find_aggregation_transform(config, "agg-node-1", "my_aggregation")

    def test_error_lists_available_transforms(self) -> None:
        """Error message lists available transform node IDs."""
        t = _make_batch_transform(node_id="other")
        config = _make_config(transforms=[t])

        with pytest.raises(OrchestrationInvariantError, match="other"):
            find_aggregation_transform(config, "agg-node-1", "batch1")

    def test_first_matching_transform_returned(self) -> None:
        """If multiple transforms match (shouldn't happen), first wins."""
        t1 = _make_batch_transform(node_id="agg-node-1")
        t2 = _make_batch_transform(node_id="agg-node-1")
        config = _make_config(transforms=[t1, t2])

        result_transform, result_node_id = find_aggregation_transform(config, "agg-node-1", "batch1")

        assert result_transform is t1
        assert result_node_id == NodeID("agg-node-1")


# =============================================================================
# handle_incomplete_batches
# =============================================================================


class TestHandleIncompleteBatches:
    """Tests for crash recovery of incomplete batches."""

    def test_executing_batch_marked_failed_then_retried(self) -> None:
        """EXECUTING batch (crash interrupted) -> failed -> retried."""
        batch = _make_batch(
            "batch-123",
            BatchStatus.EXECUTING,
            trigger_type=TriggerType.COUNT,
            trigger_reason="count=2",
            aggregation_state_id="state-123",
        )

        retry_batch = _make_batch("batch-123-retry", BatchStatus.DRAFT)

        recorder = _make_execution()
        recorder.get_incomplete_batches.return_value = [batch]
        recorder.retry_batch.return_value = retry_batch

        mapping = handle_incomplete_batches(recorder, "run-1")

        recorder.complete_batch.assert_called_once_with(
            "batch-123",
            BatchStatus.FAILED,
            trigger_type=TriggerType.COUNT,
            trigger_reason="count=2",
            state_id="state-123",
        )
        recorder.update_batch_status.assert_not_called()
        recorder.retry_batch.assert_called_once_with("batch-123")
        assert mapping == {"batch-123": "batch-123-retry"}

    def test_failed_batch_retried(self) -> None:
        """FAILED batch is retried directly."""
        batch = _make_batch("batch-456", BatchStatus.FAILED)

        retry_batch = _make_batch("batch-456-retry", BatchStatus.DRAFT)

        recorder = _make_execution()
        recorder.get_incomplete_batches.return_value = [batch]
        recorder.retry_batch.return_value = retry_batch

        mapping = handle_incomplete_batches(recorder, "run-1")

        recorder.complete_batch.assert_not_called()
        recorder.update_batch_status.assert_not_called()
        recorder.retry_batch.assert_called_once_with("batch-456")
        assert mapping == {"batch-456": "batch-456-retry"}

    def test_draft_batch_left_alone(self) -> None:
        """DRAFT batch continues collection — no action taken."""
        batch = _make_batch("batch-789", BatchStatus.DRAFT)

        recorder = _make_execution()
        recorder.get_incomplete_batches.return_value = [batch]

        mapping = handle_incomplete_batches(recorder, "run-1")

        recorder.complete_batch.assert_not_called()
        recorder.update_batch_status.assert_not_called()
        recorder.retry_batch.assert_not_called()
        assert mapping == {}

    def test_no_incomplete_batches(self) -> None:
        """No incomplete batches means no action."""
        recorder = _make_execution()
        recorder.get_incomplete_batches.return_value = []

        mapping = handle_incomplete_batches(recorder, "run-1")

        recorder.complete_batch.assert_not_called()
        recorder.update_batch_status.assert_not_called()
        recorder.retry_batch.assert_not_called()
        assert mapping == {}

    def test_multiple_batches_handled_independently(self) -> None:
        """Each batch handled according to its own status."""
        executing = _make_batch("b1", BatchStatus.EXECUTING)
        failed = _make_batch("b2", BatchStatus.FAILED)
        draft = _make_batch("b3", BatchStatus.DRAFT)

        retry_b1 = _make_batch("b1-retry", BatchStatus.DRAFT)
        retry_b2 = _make_batch("b2-retry", BatchStatus.DRAFT)

        recorder = _make_execution()
        recorder.get_incomplete_batches.return_value = [executing, failed, draft]
        recorder.retry_batch.side_effect = [retry_b1, retry_b2]

        mapping = handle_incomplete_batches(recorder, "run-1")

        recorder.complete_batch.assert_called_once_with(
            "b1",
            BatchStatus.FAILED,
            trigger_type=None,
            trigger_reason=None,
            state_id=None,
        )
        recorder.update_batch_status.assert_not_called()
        assert recorder.retry_batch.call_count == 2
        assert mapping == {"b1": "b1-retry", "b2": "b2-retry"}


# =============================================================================
# check_aggregation_timeouts
# =============================================================================


class TestCheckAggregationTimeouts:
    """Tests for pre-row aggregation timeout checks."""

    def test_no_aggregation_settings_returns_zero_result(self) -> None:
        """No aggregation settings means nothing to check."""
        config = _make_config(aggregation_settings={})
        processor = _make_processor()
        pending = _make_pending()

        result = check_aggregation_timeouts(
            config=config,
            processor=processor,
            ctx=_make_context(),
            pending_tokens=pending,
        )

        assert result.rows_succeeded == 0
        assert result.rows_failed == 0

    def test_no_flush_needed_returns_zero(self) -> None:
        """Timeout check says no flush needed — nothing happens."""
        config = _make_config(aggregation_settings={"agg-1": _make_agg_settings()})
        processor = _make_processor()
        processor.check_aggregation_timeout.return_value = (False, None)
        pending = _make_pending()

        result = check_aggregation_timeouts(
            config=config,
            processor=processor,
            ctx=_make_context(),
            pending_tokens=pending,
        )

        assert result.rows_succeeded == 0

    def test_count_trigger_skipped(self) -> None:
        """Count triggers are handled in buffer_row — skip in pre-row check."""
        config = _make_config(aggregation_settings={"agg-1": _make_agg_settings()})
        processor = _make_processor()
        processor.check_aggregation_timeout.return_value = (True, TriggerType.COUNT)
        pending = _make_pending()

        result = check_aggregation_timeouts(
            config=config,
            processor=processor,
            ctx=_make_context(),
            pending_tokens=pending,
        )

        assert result.rows_succeeded == 0

    def test_condition_trigger_flushes_pre_row(self) -> None:
        """Condition triggers that are time-based must flush before next row.

        P1-2026-02-05: Condition triggers like 'batch_age_seconds >= 5' can
        become true between rows. They must be treated like timeout triggers
        for pre-row flush, and the actual trigger_type must be passed through
        (not hardcoded as TIMEOUT).
        """
        token = make_token_info()
        completed = _make_result(TerminalOutcome.SUCCESS, TerminalPath.DEFAULT_FLOW, token=token, sink_name="output")

        agg_transform = _make_batch_transform(node_id="agg-1")
        config = _make_config(
            transforms=[agg_transform],
            aggregation_settings={"agg-1": _make_agg_settings()},
        )
        processor = _make_processor()
        processor.check_aggregation_timeout.return_value = (True, TriggerType.CONDITION)
        processor.get_aggregation_buffer_count.return_value = 3
        processor.handle_timeout_flush.return_value = ([completed], [])

        pending = _make_pending()
        lookup: dict[str, AggNodeEntry] = {"agg-1": AggNodeEntry(transform=agg_transform, node_id=NodeID("agg-1"))}

        result = check_aggregation_timeouts(
            config=config,
            processor=processor,
            ctx=_make_context(),
            pending_tokens=pending,
            agg_transform_lookup=lookup,
        )

        # Condition trigger should flush (not be skipped)
        assert result.rows_succeeded == 1
        assert len(pending["output"]) == 1

        # Verify actual trigger_type is passed (not hardcoded TIMEOUT)
        call_kwargs = processor.handle_timeout_flush.call_args.kwargs
        assert call_kwargs["trigger_type"] == TriggerType.CONDITION

    def test_empty_buffer_skipped(self) -> None:
        """Timeout fires but buffer is empty — nothing to flush."""
        config = _make_config(aggregation_settings={"agg-1": _make_agg_settings()})
        processor = _make_processor()
        processor.check_aggregation_timeout.return_value = (True, TriggerType.TIMEOUT)
        processor.get_aggregation_buffer_count.return_value = 0
        pending = _make_pending()

        result = check_aggregation_timeouts(
            config=config,
            processor=processor,
            ctx=_make_context(),
            pending_tokens=pending,
        )

        assert result.rows_succeeded == 0
        processor.handle_timeout_flush.assert_not_called()

    def test_timeout_flush_completed_results(self) -> None:
        """Timeout flush produces completed tokens routed to sink."""
        token = make_token_info()
        completed = _make_result(TerminalOutcome.SUCCESS, TerminalPath.DEFAULT_FLOW, token=token, sink_name="output")

        agg_transform = _make_batch_transform(node_id="agg-1")
        config = _make_config(
            transforms=[agg_transform],
            aggregation_settings={"agg-1": _make_agg_settings()},
        )
        processor = _make_processor()
        processor.check_aggregation_timeout.return_value = (True, TriggerType.TIMEOUT)
        processor.get_aggregation_buffer_count.return_value = 5
        processor.handle_timeout_flush.return_value = ([completed], [])

        pending = _make_pending()
        lookup: dict[str, AggNodeEntry] = {"agg-1": AggNodeEntry(transform=agg_transform, node_id=NodeID("agg-1"))}

        result = check_aggregation_timeouts(
            config=config,
            processor=processor,
            ctx=_make_context(),
            pending_tokens=pending,
            agg_transform_lookup=lookup,
        )

        assert result.rows_succeeded == 1
        assert len(pending["output"]) == 1

    def test_timeout_flush_failed_results(self) -> None:
        """Failed results from flush increment failed counter."""
        failed = _make_result(TerminalOutcome.FAILURE, TerminalPath.UNROUTED)

        agg_transform = _make_batch_transform(node_id="agg-1")
        config = _make_config(
            transforms=[agg_transform],
            aggregation_settings={"agg-1": _make_agg_settings()},
        )
        processor = _make_processor()
        processor.check_aggregation_timeout.return_value = (True, TriggerType.TIMEOUT)
        processor.get_aggregation_buffer_count.return_value = 3
        processor.handle_timeout_flush.return_value = ([failed], [])

        pending = _make_pending()
        lookup: dict[str, AggNodeEntry] = {"agg-1": AggNodeEntry(transform=agg_transform, node_id=NodeID("agg-1"))}

        result = check_aggregation_timeouts(
            config=config,
            processor=processor,
            ctx=_make_context(),
            pending_tokens=pending,
            agg_transform_lookup=lookup,
        )

        assert result.rows_failed == 1
        assert result.rows_succeeded == 0

    def test_work_items_continue_processing(self) -> None:
        """Work items from flush continue through remaining transforms."""
        work_token = make_token_info()
        work_item = _make_work_item(token=work_token, current_node_id=NodeID("continue-node"))
        downstream_result = _make_result(TerminalOutcome.SUCCESS, TerminalPath.DEFAULT_FLOW, token=work_token, sink_name="output")

        agg_transform = _make_batch_transform(node_id="agg-1")
        config = _make_config(
            transforms=[agg_transform, _TransformPlaceholder("continue-node")],
            aggregation_settings={"agg-1": _make_agg_settings()},
        )
        processor = _make_processor()
        processor.check_aggregation_timeout.return_value = (True, TriggerType.TIMEOUT)
        processor.get_aggregation_buffer_count.return_value = 2
        processor.handle_timeout_flush.return_value = ([], [work_item])
        processor.process_token.return_value = [downstream_result]

        pending = _make_pending()
        lookup: dict[str, AggNodeEntry] = {"agg-1": AggNodeEntry(transform=agg_transform, node_id=NodeID("agg-1"))}

        result = check_aggregation_timeouts(
            config=config,
            processor=processor,
            ctx=_make_context(),
            pending_tokens=pending,
            agg_transform_lookup=lookup,
        )

        assert result.rows_succeeded == 1
        processor.process_token.assert_called_once()
        assert processor.process_token.call_args.kwargs["current_node_id"] == NodeID("continue-node")

    def test_work_items_with_coalesce_node(self) -> None:
        """Work items can carry an explicit coalesce node continuation."""
        work_item = _make_work_item(
            current_node_id=NodeID("continue-node"),
            coalesce_node_id=NodeID("coalesce::merge"),
            coalesce_name="merge",
        )

        agg_transform = _make_batch_transform(node_id="agg-1")
        config = _make_config(
            transforms=[agg_transform, _TransformPlaceholder("continue-node"), _TransformPlaceholder("coalesce::merge")],
            aggregation_settings={"agg-1": _make_agg_settings()},
        )
        processor = _make_processor()
        processor.check_aggregation_timeout.return_value = (True, TriggerType.TIMEOUT)
        processor.get_aggregation_buffer_count.return_value = 1
        processor.handle_timeout_flush.return_value = ([], [work_item])
        processor.process_token.return_value = []

        pending = _make_pending()
        lookup: dict[str, AggNodeEntry] = {"agg-1": AggNodeEntry(transform=agg_transform, node_id=NodeID("agg-1"))}

        check_aggregation_timeouts(
            config=config,
            processor=processor,
            ctx=_make_context(),
            pending_tokens=pending,
            agg_transform_lookup=lookup,
        )

        assert processor.process_token.call_args.kwargs["current_node_id"] == NodeID("continue-node")
        assert processor.process_token.call_args.kwargs["coalesce_node_id"] == NodeID("coalesce::merge")
        assert processor.process_token.call_args.kwargs["coalesce_name"] == "merge"

    def test_downstream_routed_outcome(self) -> None:
        """ROUTED outcome from downstream is tracked."""
        work_item = _make_work_item()
        routed = _make_result(TerminalOutcome.SUCCESS, TerminalPath.GATE_ROUTED, sink_name="risk_sink")

        agg_transform = _make_batch_transform(node_id="agg-1")
        config = _make_config(
            transforms=[agg_transform],
            aggregation_settings={"agg-1": _make_agg_settings()},
        )
        processor = _make_processor()
        processor.check_aggregation_timeout.return_value = (True, TriggerType.TIMEOUT)
        processor.get_aggregation_buffer_count.return_value = 1
        processor.handle_timeout_flush.return_value = ([], [work_item])
        processor.process_token.return_value = [routed]

        pending: dict[str, list[tuple[TokenInfo, PendingOutcome | None]]] = {"output": [], "risk_sink": []}
        lookup: dict[str, AggNodeEntry] = {"agg-1": AggNodeEntry(transform=agg_transform, node_id=NodeID("agg-1"))}

        result = check_aggregation_timeouts(
            config=config,
            processor=processor,
            ctx=_make_context(),
            pending_tokens=pending,
            agg_transform_lookup=lookup,
        )

        assert result.rows_routed_success == 1
        assert result.rows_succeeded == 1
        assert result.rows_routed_failure == 0
        assert result.routed_destinations == {"risk_sink": 1}
        assert len(pending["risk_sink"]) == 1

    def test_downstream_quarantined_outcome(self) -> None:
        """QUARANTINED outcome from downstream is counted."""
        work_item = _make_work_item()
        quarantined = _make_result(TerminalOutcome.FAILURE, TerminalPath.QUARANTINED_AT_SOURCE)

        agg_transform = _make_batch_transform(node_id="agg-1")
        config = _make_config(
            transforms=[agg_transform],
            aggregation_settings={"agg-1": _make_agg_settings()},
        )
        processor = _make_processor()
        processor.check_aggregation_timeout.return_value = (True, TriggerType.TIMEOUT)
        processor.get_aggregation_buffer_count.return_value = 1
        processor.handle_timeout_flush.return_value = ([], [work_item])
        processor.process_token.return_value = [quarantined]

        pending = _make_pending()
        lookup: dict[str, AggNodeEntry] = {"agg-1": AggNodeEntry(transform=agg_transform, node_id=NodeID("agg-1"))}

        result = check_aggregation_timeouts(
            config=config,
            processor=processor,
            ctx=_make_context(),
            pending_tokens=pending,
            agg_transform_lookup=lookup,
        )

        assert result.rows_quarantined == 1

    def test_downstream_coalesced_outcome(self) -> None:
        """COALESCED outcome increments both coalesced and succeeded."""
        work_item = _make_work_item()
        coalesced = _make_result(TerminalOutcome.SUCCESS, TerminalPath.COALESCED, sink_name="output")

        agg_transform = _make_batch_transform(node_id="agg-1")
        config = _make_config(
            transforms=[agg_transform],
            aggregation_settings={"agg-1": _make_agg_settings()},
        )
        processor = _make_processor()
        processor.check_aggregation_timeout.return_value = (True, TriggerType.TIMEOUT)
        processor.get_aggregation_buffer_count.return_value = 1
        processor.handle_timeout_flush.return_value = ([], [work_item])
        processor.process_token.return_value = [coalesced]

        pending = _make_pending()
        lookup: dict[str, AggNodeEntry] = {"agg-1": AggNodeEntry(transform=agg_transform, node_id=NodeID("agg-1"))}

        result = check_aggregation_timeouts(
            config=config,
            processor=processor,
            ctx=_make_context(),
            pending_tokens=pending,
            agg_transform_lookup=lookup,
        )

        assert result.rows_coalesced == 1
        assert result.rows_succeeded == 1

    def test_downstream_failed_in_timeout(self) -> None:
        """FAILED downstream outcome from work items in timeout check."""
        work_item = _make_work_item()
        failed = _make_result(TerminalOutcome.FAILURE, TerminalPath.UNROUTED)

        agg_transform = _make_batch_transform(node_id="agg-1")
        config = _make_config(
            transforms=[agg_transform],
            aggregation_settings={"agg-1": _make_agg_settings()},
        )
        processor = _make_processor()
        processor.check_aggregation_timeout.return_value = (True, TriggerType.TIMEOUT)
        processor.get_aggregation_buffer_count.return_value = 1
        processor.handle_timeout_flush.return_value = ([], [work_item])
        processor.process_token.return_value = [failed]

        pending = _make_pending()
        lookup: dict[str, AggNodeEntry] = {"agg-1": AggNodeEntry(transform=agg_transform, node_id=NodeID("agg-1"))}

        result = check_aggregation_timeouts(
            config=config,
            processor=processor,
            ctx=_make_context(),
            pending_tokens=pending,
            agg_transform_lookup=lookup,
        )

        assert result.rows_failed == 1

    def test_downstream_completed_branch_fallback_in_timeout(self) -> None:
        """COMPLETED work item with unknown branch routes to sink_name from result."""
        token = TokenInfo(row_id="row-1", token_id="tok-1", row_data=make_row({}), branch_name="unknown")
        work_item = _make_work_item(token=token)
        completed = _make_result(TerminalOutcome.SUCCESS, TerminalPath.DEFAULT_FLOW, token=token, sink_name="output")

        agg_transform = _make_batch_transform(node_id="agg-1")
        config = _make_config(
            transforms=[agg_transform],
            aggregation_settings={"agg-1": _make_agg_settings()},
        )
        processor = _make_processor()
        processor.check_aggregation_timeout.return_value = (True, TriggerType.TIMEOUT)
        processor.get_aggregation_buffer_count.return_value = 1
        processor.handle_timeout_flush.return_value = ([], [work_item])
        processor.process_token.return_value = [completed]

        pending = _make_pending()
        lookup: dict[str, AggNodeEntry] = {"agg-1": AggNodeEntry(transform=agg_transform, node_id=NodeID("agg-1"))}

        result = check_aggregation_timeouts(
            config=config,
            processor=processor,
            ctx=_make_context(),
            pending_tokens=pending,
            agg_transform_lookup=lookup,
        )

        assert result.rows_succeeded == 1
        assert len(pending["output"]) == 1

    def test_downstream_forked_expanded_buffered(self) -> None:
        """FORKED, EXPANDED, BUFFERED outcomes each tracked separately."""
        work_item = _make_work_item()
        outcomes = [
            _make_result(TerminalOutcome.TRANSIENT, TerminalPath.FORK_PARENT),
            _make_result(TerminalOutcome.TRANSIENT, TerminalPath.EXPAND_PARENT),
            _make_result(None, TerminalPath.BUFFERED),
        ]

        agg_transform = _make_batch_transform(node_id="agg-1")
        config = _make_config(
            transforms=[agg_transform],
            aggregation_settings={"agg-1": _make_agg_settings()},
        )
        processor = _make_processor()
        processor.check_aggregation_timeout.return_value = (True, TriggerType.TIMEOUT)
        processor.get_aggregation_buffer_count.return_value = 1
        processor.handle_timeout_flush.return_value = ([], [work_item])
        processor.process_token.return_value = outcomes

        pending = _make_pending()
        lookup: dict[str, AggNodeEntry] = {"agg-1": AggNodeEntry(transform=agg_transform, node_id=NodeID("agg-1"))}

        result = check_aggregation_timeouts(
            config=config,
            processor=processor,
            ctx=_make_context(),
            pending_tokens=pending,
            agg_transform_lookup=lookup,
        )

        assert result.rows_forked == 1
        assert result.rows_expanded == 1
        assert result.rows_buffered == 1

    def test_completed_result_branch_fallback_in_timeout(self) -> None:
        """Completed result with branch not in pending routes to sink_name from result."""
        token = TokenInfo(row_id="row-1", token_id="tok-1", row_data=make_row({}), branch_name="missing_sink")
        completed = _make_result(TerminalOutcome.SUCCESS, TerminalPath.DEFAULT_FLOW, token=token, sink_name="output")

        agg_transform = _make_batch_transform(node_id="agg-1")
        config = _make_config(
            transforms=[agg_transform],
            aggregation_settings={"agg-1": _make_agg_settings()},
        )
        processor = _make_processor()
        processor.check_aggregation_timeout.return_value = (True, TriggerType.TIMEOUT)
        processor.get_aggregation_buffer_count.return_value = 1
        processor.handle_timeout_flush.return_value = ([completed], [])

        pending = _make_pending()
        lookup: dict[str, AggNodeEntry] = {"agg-1": AggNodeEntry(transform=agg_transform, node_id=NodeID("agg-1"))}

        result = check_aggregation_timeouts(
            config=config,
            processor=processor,
            ctx=_make_context(),
            pending_tokens=pending,
            agg_transform_lookup=lookup,
        )

        assert result.rows_succeeded == 1
        assert len(pending["output"]) == 1

    def test_fallback_lookup_when_no_cache(self) -> None:
        """Without agg_transform_lookup, find_aggregation_transform is called."""
        agg_transform = _make_batch_transform(node_id="agg-1")
        config = _make_config(
            transforms=[agg_transform],
            aggregation_settings={"agg-1": _make_agg_settings()},
        )
        processor = _make_processor()
        processor.check_aggregation_timeout.return_value = (True, TriggerType.TIMEOUT)
        processor.get_aggregation_buffer_count.return_value = 1
        processor.handle_timeout_flush.return_value = ([], [])

        pending = _make_pending()

        # No lookup passed — function should find the transform itself
        check_aggregation_timeouts(
            config=config,
            processor=processor,
            ctx=_make_context(),
            pending_tokens=pending,
            agg_transform_lookup=None,
        )

        processor.handle_timeout_flush.assert_called_once()


# =============================================================================
# flush_remaining_aggregation_buffers
# =============================================================================


class TestFlushRemainingAggregationBuffers:
    """Tests for end-of-source aggregation buffer flush."""

    def test_empty_buffer_skipped(self) -> None:
        """Aggregation with empty buffer is skipped."""
        config = _make_config(aggregation_settings={"agg-1": _make_agg_settings()})
        processor = _make_processor()
        processor.get_aggregation_buffer_count.return_value = 0
        pending = _make_pending()

        result = flush_remaining_aggregation_buffers(
            config=config,
            processor=processor,
            ctx=_make_context(),
            pending_tokens=pending,
        )

        assert result.rows_succeeded == 0
        processor.handle_timeout_flush.assert_not_called()

    def test_flush_completed_results(self) -> None:
        """Completed results from flush go to sink."""
        token = make_token_info()
        completed = _make_result(TerminalOutcome.SUCCESS, TerminalPath.DEFAULT_FLOW, token=token, sink_name="output")

        agg_transform = _make_batch_transform(node_id="agg-1")
        config = _make_config(
            transforms=[agg_transform],
            aggregation_settings={"agg-1": _make_agg_settings()},
        )
        processor = _make_processor()
        processor.get_aggregation_buffer_count.return_value = 3
        processor.handle_timeout_flush.return_value = ([completed], [])

        pending = _make_pending()

        result = flush_remaining_aggregation_buffers(
            config=config,
            processor=processor,
            ctx=_make_context(),
            pending_tokens=pending,
        )

        assert result.rows_succeeded == 1
        assert len(pending["output"]) == 1

    def test_uses_end_of_source_trigger(self) -> None:
        """Flush uses END_OF_SOURCE trigger type."""
        agg_transform = _make_batch_transform(node_id="agg-1")
        config = _make_config(
            transforms=[agg_transform],
            aggregation_settings={"agg-1": _make_agg_settings()},
        )
        processor = _make_processor()
        processor.get_aggregation_buffer_count.return_value = 1
        processor.handle_timeout_flush.return_value = ([], [])

        pending = _make_pending()

        flush_remaining_aggregation_buffers(
            config=config,
            processor=processor,
            ctx=_make_context(),
            pending_tokens=pending,
        )

        call_kwargs = processor.handle_timeout_flush.call_args.kwargs
        assert call_kwargs["trigger_type"] == TriggerType.END_OF_SOURCE

    def test_work_items_continue_downstream(self) -> None:
        """Work items from flush continue through remaining transforms."""
        work_token = make_token_info()
        work_item = _make_work_item(token=work_token, current_node_id=NodeID("continue-node"))
        downstream = _make_result(TerminalOutcome.SUCCESS, TerminalPath.DEFAULT_FLOW, token=work_token, sink_name="output")

        agg_transform = _make_batch_transform(node_id="agg-1")
        config = _make_config(
            transforms=[agg_transform, _TransformPlaceholder("continue-node")],
            aggregation_settings={"agg-1": _make_agg_settings()},
        )
        processor = _make_processor()
        processor.get_aggregation_buffer_count.return_value = 1
        processor.handle_timeout_flush.return_value = ([], [work_item])
        processor.process_token.return_value = [downstream]

        pending = _make_pending()

        result = flush_remaining_aggregation_buffers(
            config=config,
            processor=processor,
            ctx=_make_context(),
            pending_tokens=pending,
        )

        assert result.rows_succeeded == 1

    def test_downstream_routed_tokens_counted(self) -> None:
        """ROUTED downstream outcome is counted correctly."""
        work_item = _make_work_item()
        routed = _make_result(TerminalOutcome.SUCCESS, TerminalPath.GATE_ROUTED, sink_name="risk")

        agg_transform = _make_batch_transform(node_id="agg-1")
        config = _make_config(
            transforms=[agg_transform],
            aggregation_settings={"agg-1": _make_agg_settings()},
        )
        processor = _make_processor()
        processor.get_aggregation_buffer_count.return_value = 1
        processor.handle_timeout_flush.return_value = ([], [work_item])
        processor.process_token.return_value = [routed]

        pending: dict[str, list[tuple[TokenInfo, PendingOutcome | None]]] = {"output": [], "risk": []}

        result = flush_remaining_aggregation_buffers(
            config=config,
            processor=processor,
            ctx=_make_context(),
            pending_tokens=pending,
        )

        assert result.rows_routed_success == 1
        assert result.rows_succeeded == 1
        assert result.rows_routed_failure == 0

    def test_downstream_coalesced_tokens_counted(self) -> None:
        """COALESCED downstream outcome increments both counters."""
        work_item = _make_work_item()
        coalesced = _make_result(TerminalOutcome.SUCCESS, TerminalPath.COALESCED, sink_name="output")

        agg_transform = _make_batch_transform(node_id="agg-1")
        config = _make_config(
            transforms=[agg_transform],
            aggregation_settings={"agg-1": _make_agg_settings()},
        )
        processor = _make_processor()
        processor.get_aggregation_buffer_count.return_value = 1
        processor.handle_timeout_flush.return_value = ([], [work_item])
        processor.process_token.return_value = [coalesced]

        pending = _make_pending()

        result = flush_remaining_aggregation_buffers(
            config=config,
            processor=processor,
            ctx=_make_context(),
            pending_tokens=pending,
        )

        assert result.rows_coalesced == 1
        assert result.rows_succeeded == 1

    def test_branch_routing_for_completed_tokens(self) -> None:
        """Completed tokens route via result.sink_name, not branch_name."""
        token = TokenInfo(row_id="row-1", token_id="tok-1", row_data=make_row({}), branch_name="path_a")
        completed = _make_result(TerminalOutcome.SUCCESS, TerminalPath.DEFAULT_FLOW, token=token, sink_name="output")

        agg_transform = _make_batch_transform(node_id="agg-1")
        config = _make_config(
            transforms=[agg_transform],
            aggregation_settings={"agg-1": _make_agg_settings()},
        )
        processor = _make_processor()
        processor.get_aggregation_buffer_count.return_value = 1
        processor.handle_timeout_flush.return_value = ([completed], [])

        pending: dict[str, list[tuple[TokenInfo, PendingOutcome | None]]] = {"output": [], "path_a": []}

        result = flush_remaining_aggregation_buffers(
            config=config,
            processor=processor,
            ctx=_make_context(),
            pending_tokens=pending,
        )

        assert result.rows_succeeded == 1
        assert len(pending["path_a"]) == 0
        assert len(pending["output"]) == 1

    def test_downstream_failed_in_flush(self) -> None:
        """FAILED outcome from downstream work items counted in flush."""
        work_item = _make_work_item()
        failed = _make_result(TerminalOutcome.FAILURE, TerminalPath.UNROUTED)

        agg_transform = _make_batch_transform(node_id="agg-1")
        config = _make_config(
            transforms=[agg_transform],
            aggregation_settings={"agg-1": _make_agg_settings()},
        )
        processor = _make_processor()
        processor.get_aggregation_buffer_count.return_value = 1
        processor.handle_timeout_flush.return_value = ([], [work_item])
        processor.process_token.return_value = [failed]

        pending = _make_pending()

        result = flush_remaining_aggregation_buffers(
            config=config,
            processor=processor,
            ctx=_make_context(),
            pending_tokens=pending,
        )

        assert result.rows_failed == 1

    def test_downstream_completed_branch_fallback_in_flush(self) -> None:
        """COMPLETED work item with unknown branch routes to sink_name from result."""
        token = TokenInfo(row_id="row-1", token_id="tok-1", row_data=make_row({}), branch_name="unknown")
        work_item = _make_work_item(token=token)
        completed = _make_result(TerminalOutcome.SUCCESS, TerminalPath.DEFAULT_FLOW, token=token, sink_name="output")

        agg_transform = _make_batch_transform(node_id="agg-1")
        config = _make_config(
            transforms=[agg_transform],
            aggregation_settings={"agg-1": _make_agg_settings()},
        )
        processor = _make_processor()
        processor.get_aggregation_buffer_count.return_value = 1
        processor.handle_timeout_flush.return_value = ([], [work_item])
        processor.process_token.return_value = [completed]

        pending = _make_pending()

        result = flush_remaining_aggregation_buffers(
            config=config,
            processor=processor,
            ctx=_make_context(),
            pending_tokens=pending,
        )

        assert result.rows_succeeded == 1
        assert len(pending["output"]) == 1

    def test_downstream_quarantined_in_flush(self) -> None:
        """QUARANTINED outcome from downstream work items counted in flush."""
        work_item = _make_work_item()
        quarantined = _make_result(TerminalOutcome.FAILURE, TerminalPath.QUARANTINED_AT_SOURCE)

        agg_transform = _make_batch_transform(node_id="agg-1")
        config = _make_config(
            transforms=[agg_transform],
            aggregation_settings={"agg-1": _make_agg_settings()},
        )
        processor = _make_processor()
        processor.get_aggregation_buffer_count.return_value = 1
        processor.handle_timeout_flush.return_value = ([], [work_item])
        processor.process_token.return_value = [quarantined]

        pending = _make_pending()

        result = flush_remaining_aggregation_buffers(
            config=config,
            processor=processor,
            ctx=_make_context(),
            pending_tokens=pending,
        )

        assert result.rows_quarantined == 1

    def test_downstream_forked_expanded_buffered_in_flush(self) -> None:
        """FORKED, EXPANDED, BUFFERED outcomes from work items tracked in flush."""
        work_item = _make_work_item()
        outcomes = [
            _make_result(TerminalOutcome.TRANSIENT, TerminalPath.FORK_PARENT),
            _make_result(TerminalOutcome.TRANSIENT, TerminalPath.EXPAND_PARENT),
            _make_result(None, TerminalPath.BUFFERED),
        ]

        agg_transform = _make_batch_transform(node_id="agg-1")
        config = _make_config(
            transforms=[agg_transform],
            aggregation_settings={"agg-1": _make_agg_settings()},
        )
        processor = _make_processor()
        processor.get_aggregation_buffer_count.return_value = 1
        processor.handle_timeout_flush.return_value = ([], [work_item])
        processor.process_token.return_value = outcomes

        pending = _make_pending()

        result = flush_remaining_aggregation_buffers(
            config=config,
            processor=processor,
            ctx=_make_context(),
            pending_tokens=pending,
        )

        assert result.rows_forked == 1
        assert result.rows_expanded == 1
        assert result.rows_buffered == 1

    def test_work_item_with_coalesce_node_in_flush(self) -> None:
        """Work items with coalesce_node_id preserve continuation metadata in flush."""
        work_item = _make_work_item(
            current_node_id=NodeID("continue-node"),
            coalesce_node_id=NodeID("coalesce::merge"),
            coalesce_name="merge",
        )

        agg_transform = _make_batch_transform(node_id="agg-1")
        config = _make_config(
            transforms=[agg_transform, _TransformPlaceholder("continue-node"), _TransformPlaceholder("coalesce::merge")],
            aggregation_settings={"agg-1": _make_agg_settings()},
        )
        processor = _make_processor()
        processor.get_aggregation_buffer_count.return_value = 1
        processor.handle_timeout_flush.return_value = ([], [work_item])
        processor.process_token.return_value = []

        pending = _make_pending()

        flush_remaining_aggregation_buffers(
            config=config,
            processor=processor,
            ctx=_make_context(),
            pending_tokens=pending,
        )

        assert processor.process_token.call_args.kwargs["current_node_id"] == NodeID("continue-node")
        assert processor.process_token.call_args.kwargs["coalesce_node_id"] == NodeID("coalesce::merge")

    def test_completed_result_branch_fallback_to_sink_name(self) -> None:
        """Completed result with branch not in pending routes to sink_name from result."""
        token = TokenInfo(row_id="row-1", token_id="tok-1", row_data=make_row({}), branch_name="missing")
        completed = _make_result(TerminalOutcome.SUCCESS, TerminalPath.DEFAULT_FLOW, token=token, sink_name="output")

        agg_transform = _make_batch_transform(node_id="agg-1")
        config = _make_config(
            transforms=[agg_transform],
            aggregation_settings={"agg-1": _make_agg_settings()},
        )
        processor = _make_processor()
        processor.get_aggregation_buffer_count.return_value = 1
        processor.handle_timeout_flush.return_value = ([completed], [])

        pending = _make_pending()

        result = flush_remaining_aggregation_buffers(
            config=config,
            processor=processor,
            ctx=_make_context(),
            pending_tokens=pending,
        )

        assert result.rows_succeeded == 1
        assert len(pending["output"]) == 1

    def test_branch_routing_falls_back_to_sink_name(self) -> None:
        """Branch name not in pending_tokens routes to sink_name from result."""
        token = TokenInfo(row_id="row-1", token_id="tok-1", row_data=make_row({}), branch_name="nonexistent")
        completed = _make_result(TerminalOutcome.SUCCESS, TerminalPath.DEFAULT_FLOW, token=token, sink_name="output")

        agg_transform = _make_batch_transform(node_id="agg-1")
        config = _make_config(
            transforms=[agg_transform],
            aggregation_settings={"agg-1": _make_agg_settings()},
        )
        processor = _make_processor()
        processor.get_aggregation_buffer_count.return_value = 1
        processor.handle_timeout_flush.return_value = ([completed], [])

        pending = _make_pending()

        result = flush_remaining_aggregation_buffers(
            config=config,
            processor=processor,
            ctx=_make_context(),
            pending_tokens=pending,
        )

        assert result.rows_succeeded == 1
        assert len(pending["output"]) == 1
