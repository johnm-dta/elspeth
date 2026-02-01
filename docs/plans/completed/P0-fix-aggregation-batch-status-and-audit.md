# Implementation Plan: Fix Aggregation Batch Status and Audit Trail

**Bug:** P0-2026-01-19-aggregation-batch-status-and-audit-missing.md
**Estimated Time:** 3-4 hours
**Complexity:** Medium-High
**Risk:** Medium (changes to core execution path)

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

## Summary

Three interconnected issues with aggregation batch handling:
1. **Missing audit trail** - Batch flush bypasses `TransformExecutor`, no `node_state` recorded
2. **Batch status stuck in draft** - Never transitions through `executing → completed/failed`
3. **Broken recovery** - Recovery expects `executing`/`failed` batches but they're stuck in `draft`

## Current Behavior

```python
# processor.py:199 - Direct call bypasses audit recording
result = transform.process(buffered_rows, ctx)  # No node_state!

# Batches created in draft status, never updated
batch = recorder.create_batch(...)  # status = "draft"
# ... flush happens ...
# batch.status is STILL "draft" - no update_batch_status() call
```

## Architecture Understanding

```
┌─────────────────────────────────────────────────────────────────────┐
│                     TransformExecutor Flow                          │
│  (for single-row transforms - CORRECT)                              │
│                                                                      │
│  1. begin_node_state()     → records input                          │
│  2. transform.process()    → timed execution                        │
│  3. complete_node_state()  → records output, duration               │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                     Aggregation Flush Flow                           │
│  (CURRENT - missing audit)                                          │
│                                                                      │
│  1. AggregationExecutor.buffer_row()  → creates batch, adds member  │
│  2. should_flush() → true                                           │
│  3. flush_buffer() → returns rows/tokens                            │
│  4. transform.process(buffered_rows)  ← NO AUDIT WRAPPER!           │
│  5. (batch status never updated)                                    │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                     Aggregation Flush Flow                           │
│  (FIXED - with audit)                                               │
│                                                                      │
│  1. AggregationExecutor.buffer_row()  → creates batch, adds member  │
│  2. should_flush() → true                                           │
│  3. execute_flush():                                                │
│     a. update_batch_status("executing", trigger_reason)             │
│     b. begin_node_state() for flush operation                       │
│     c. transform.process(buffered_rows)                             │
│     d. complete_node_state() with output hash                       │
│     e. complete_batch("completed"/"failed", state_id)               │
│  4. Reset batch_id for next batch                                   │
└─────────────────────────────────────────────────────────────────────┘
```

## Implementation Steps

### Step 1: Add `execute_flush()` method to AggregationExecutor

**File:** `src/elspeth/engine/executors.py`

**Location:** Add after `flush_buffer()` method (around line 840)

```python
def execute_flush(
    self,
    node_id: str,
    transform: TransformProtocol,
    ctx: PluginContext,
    step_in_pipeline: int,
    trigger_type: TriggerType,
) -> tuple[TransformResult, list[TokenInfo]]:
    """Execute a batch flush with full audit recording.

    This method:
    1. Transitions batch to "executing" with trigger reason
    2. Records node_state for the flush operation
    3. Executes the batch-aware transform
    4. Transitions batch to "completed" or "failed"
    5. Resets batch_id for next batch

    Args:
        node_id: Aggregation node ID
        transform: Batch-aware transform plugin
        ctx: Plugin context
        step_in_pipeline: Current position in DAG
        trigger_type: What triggered the flush (COUNT, TIMEOUT, END_OF_SOURCE, etc.)

    Returns:
        Tuple of (TransformResult with audit fields, list of consumed tokens)

    Raises:
        Exception: Re-raised from transform.process() after recording failure
    """
    from elspeth.core.canonical import stable_hash

    # Get batch_id - must exist if we're flushing
    batch_id = self._batch_ids.get(node_id)
    if batch_id is None:
        raise RuntimeError(f"No batch exists for node {node_id} - cannot flush")

    # Get buffered data
    buffered_rows = list(self._buffers.get(node_id, []))
    buffered_tokens = list(self._buffer_tokens.get(node_id, []))

    if not buffered_rows:
        raise RuntimeError(f"Cannot flush empty buffer for node {node_id}")

    # Compute input hash for batch (hash of all input rows)
    input_hash = stable_hash(buffered_rows)

    # Use first token for node_state (represents the batch operation)
    # The batch_members table links all consumed tokens to this batch
    representative_token = buffered_tokens[0]

    # Step 1: Transition batch to "executing"
    self._recorder.update_batch_status(
        batch_id=batch_id,
        status="executing",
        trigger_reason=trigger_type.value,
    )

    # Step 2: Begin node state for flush operation
    state = self._recorder.begin_node_state(
        token_id=representative_token.token_id,
        node_id=node_id,
        step_index=step_in_pipeline,
        input_data=buffered_rows,  # Store all input rows
        attempt=0,  # Aggregation flushes don't retry (yet)
    )

    # Step 3: Execute with timing and span
    with self._spans.transform_span(transform.name, input_hash=input_hash):
        start = time.perf_counter()
        try:
            # Type ignore: batch-aware transforms accept list[dict] at runtime
            result = transform.process(buffered_rows, ctx)  # type: ignore[arg-type]
            duration_ms = (time.perf_counter() - start) * 1000
        except Exception as e:
            duration_ms = (time.perf_counter() - start) * 1000

            # Record failure in node_state
            error: ExecutionError = {
                "exception": str(e),
                "type": type(e).__name__,
            }
            self._recorder.complete_node_state(
                state_id=state.state_id,
                status="failed",
                duration_ms=duration_ms,
                error=error,
            )

            # Transition batch to failed
            self._recorder.complete_batch(
                batch_id=batch_id,
                status="failed",
                trigger_reason=trigger_type.value,
                state_id=state.state_id,
            )

            # Reset for next batch
            self._reset_batch_state(node_id)
            raise

    # Step 4: Populate audit fields on result
    result.input_hash = input_hash
    if result.row is not None:
        result.output_hash = stable_hash(result.row)
    elif result.rows is not None:
        result.output_hash = stable_hash(result.rows)
    else:
        result.output_hash = None
    result.duration_ms = duration_ms

    # Step 5: Complete node state
    if result.status == "success":
        output_data: dict[str, Any] | list[dict[str, Any]]
        if result.row is not None:
            output_data = result.row
        else:
            assert result.rows is not None
            output_data = result.rows

        self._recorder.complete_node_state(
            state_id=state.state_id,
            status="completed",
            output_data=output_data,
            duration_ms=duration_ms,
        )

        # Transition batch to completed
        self._recorder.complete_batch(
            batch_id=batch_id,
            status="completed",
            trigger_reason=trigger_type.value,
            state_id=state.state_id,
        )
    else:
        # Transform returned error status
        error_info: ExecutionError = {
            "exception": result.error_message or "Transform returned error",
            "type": "TransformError",
        }
        self._recorder.complete_node_state(
            state_id=state.state_id,
            status="failed",
            duration_ms=duration_ms,
            error=error_info,
        )

        # Transition batch to failed
        self._recorder.complete_batch(
            batch_id=batch_id,
            status="failed",
            trigger_reason=trigger_type.value,
            state_id=state.state_id,
        )

    # Step 6: Reset for next batch and clear buffers
    self._reset_batch_state(node_id)
    self._buffers[node_id] = []
    self._buffer_tokens[node_id] = []

    return result, buffered_tokens


def _reset_batch_state(self, node_id: str) -> None:
    """Reset batch tracking state for next batch.

    Args:
        node_id: Aggregation node ID
    """
    batch_id = self._batch_ids.get(node_id)
    if batch_id is not None:
        del self._batch_ids[node_id]
        if batch_id in self._member_counts:
            del self._member_counts[batch_id]
```

### Step 2: Add required imports to executors.py

**File:** `src/elspeth/engine/executors.py`

**Location:** At the top, add to existing imports

```python
from elspeth.contracts.enums import TriggerType
```

### Step 3: Add `get_trigger_type()` method to AggregationExecutor

**File:** `src/elspeth/engine/executors.py`

**Location:** Add near `should_flush()` method

```python
def get_trigger_type(self, node_id: str) -> TriggerType | None:
    """Get the trigger type that caused the current flush condition.

    Must be called when should_flush() is True.

    Args:
        node_id: Aggregation node ID

    Returns:
        TriggerType that caused the flush, or None if no trigger evaluator
    """
    evaluator = self._trigger_evaluators.get(node_id)
    if evaluator is not None:
        return evaluator.get_trigger_type()
    return None
```

### Step 4: Modify RowProcessor to use execute_flush()

**File:** `src/elspeth/engine/processor.py`

**Location:** In `_handle_aggregation_transform()` method (around line 188-213)

**Replace the direct transform.process() call with execute_flush():**

```python
# Check if we should flush
if self._aggregation_executor.should_flush(node_id):
    # Determine trigger type
    trigger_type = self._aggregation_executor.get_trigger_type(node_id)
    if trigger_type is None:
        trigger_type = TriggerType.COUNT  # Default if no evaluator

    # Execute flush with full audit recording
    result, buffered_tokens = self._aggregation_executor.execute_flush(
        node_id=node_id,
        transform=transform,
        ctx=ctx,
        step_in_pipeline=step_in_pipeline,
        trigger_type=trigger_type,
    )

    if result.status != "success":
        return (
            RowResult(
                token=current_token,
                final_data=current_token.row_data,
                outcome=RowOutcome.FAILED,
                error=FailureInfo(
                    exception_type="TransformError",
                    message="Batch transform failed",
                ),
            ),
            child_items,
        )

    # Handle output modes (existing logic continues here...)
```

### Step 5: Add TriggerType import to processor.py

**File:** `src/elspeth/engine/processor.py`

**Location:** At the top, add to existing imports

```python
from elspeth.contracts.enums import TriggerType
```

### Step 6: Handle end-of-source flush

**File:** `src/elspeth/engine/orchestrator.py`

**Location:** In the end-of-run aggregation flush logic

Find where remaining aggregation buffers are flushed at end of source and ensure they use `execute_flush()` with `TriggerType.END_OF_SOURCE`.

```python
# When flushing remaining aggregation buffers at end of source:
result, buffered_tokens = aggregation_executor.execute_flush(
    node_id=node_id,
    transform=transform,
    ctx=ctx,
    step_in_pipeline=step_index,
    trigger_type=TriggerType.END_OF_SOURCE,
)
```

### Step 7: Update flush_buffer() to not clear buffers

**File:** `src/elspeth/engine/executors.py`

Since `execute_flush()` now manages buffer clearing, ensure the old `flush_buffer()` method is only used for getting data (if still needed) or deprecate it.

**Option A:** Keep `flush_buffer()` for backwards compatibility but mark deprecated
**Option B:** Remove `flush_buffer()` since `execute_flush()` handles everything

Recommended: Remove direct usage, keep method for tests only.

### Step 8: Add unit tests

**File:** `tests/engine/test_aggregation_audit.py` (new file)

```python
"""Tests for aggregation batch audit trail."""

import pytest

from elspeth.contracts.enums import BatchStatus, TriggerType
from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.recorder import LandscapeRecorder
from elspeth.engine.executors import AggregationExecutor, SpanFactory
from elspeth.engine.triggers import TriggerConfig


class TestAggregationFlushAudit:
    """Tests for aggregation flush audit recording."""

    def test_flush_creates_node_state(self, landscape_db: LandscapeDB):
        """Flushing a batch should create a node_state record."""
        recorder = LandscapeRecorder(landscape_db)
        run = recorder.start_run(config_hash="test", config_json={})

        executor = AggregationExecutor(
            recorder=recorder,
            span_factory=SpanFactory.null(),
            run_id=run.run_id,
        )

        # Setup: buffer some rows
        node_id = "agg_1"
        for i in range(3):
            token = TokenInfo(row_id=f"row_{i}", token_id=f"token_{i}", row_data={"x": i})
            executor.buffer_row(node_id, token)

        # Execute flush
        result, tokens = executor.execute_flush(
            node_id=node_id,
            transform=MockBatchTransform(),
            ctx=PluginContext(),
            step_in_pipeline=0,
            trigger_type=TriggerType.COUNT,
        )

        # Verify node_state was created
        states = recorder.get_node_states_for_run(run.run_id)
        assert len(states) == 1
        assert states[0].node_id == node_id
        assert states[0].status == "completed"
        assert states[0].input_data is not None  # Contains all input rows

    def test_flush_transitions_batch_status(self, landscape_db: LandscapeDB):
        """Flushing should transition batch from draft → executing → completed."""
        recorder = LandscapeRecorder(landscape_db)
        run = recorder.start_run(config_hash="test", config_json={})

        executor = AggregationExecutor(
            recorder=recorder,
            span_factory=SpanFactory.null(),
            run_id=run.run_id,
        )

        # Buffer and flush
        node_id = "agg_1"
        token = TokenInfo(row_id="row_1", token_id="token_1", row_data={"x": 1})
        executor.buffer_row(node_id, token)

        # Get batch_id before flush
        batch_id = executor._batch_ids[node_id]

        # Verify initial status is draft
        batch = recorder.get_batch(batch_id)
        assert batch.status == BatchStatus.DRAFT

        # Execute flush
        executor.execute_flush(
            node_id=node_id,
            transform=MockBatchTransform(),
            ctx=PluginContext(),
            step_in_pipeline=0,
            trigger_type=TriggerType.COUNT,
        )

        # Verify final status is completed
        batch = recorder.get_batch(batch_id)
        assert batch.status == BatchStatus.COMPLETED
        assert batch.trigger_reason == "count"
        assert batch.aggregation_state_id is not None

    def test_failed_flush_marks_batch_failed(self, landscape_db: LandscapeDB):
        """Exception during flush should mark batch as failed."""
        recorder = LandscapeRecorder(landscape_db)
        run = recorder.start_run(config_hash="test", config_json={})

        executor = AggregationExecutor(
            recorder=recorder,
            span_factory=SpanFactory.null(),
            run_id=run.run_id,
        )

        # Buffer row
        node_id = "agg_1"
        token = TokenInfo(row_id="row_1", token_id="token_1", row_data={"x": 1})
        executor.buffer_row(node_id, token)
        batch_id = executor._batch_ids[node_id]

        # Execute flush with failing transform
        with pytest.raises(RuntimeError, match="intentional"):
            executor.execute_flush(
                node_id=node_id,
                transform=FailingBatchTransform(),
                ctx=PluginContext(),
                step_in_pipeline=0,
                trigger_type=TriggerType.COUNT,
            )

        # Verify batch is marked failed
        batch = recorder.get_batch(batch_id)
        assert batch.status == BatchStatus.FAILED
        assert batch.aggregation_state_id is not None

    def test_end_of_source_trigger_recorded(self, landscape_db: LandscapeDB):
        """END_OF_SOURCE trigger reason should be recorded."""
        recorder = LandscapeRecorder(landscape_db)
        run = recorder.start_run(config_hash="test", config_json={})

        executor = AggregationExecutor(
            recorder=recorder,
            span_factory=SpanFactory.null(),
            run_id=run.run_id,
        )

        # Buffer and flush with END_OF_SOURCE
        node_id = "agg_1"
        token = TokenInfo(row_id="row_1", token_id="token_1", row_data={"x": 1})
        executor.buffer_row(node_id, token)
        batch_id = executor._batch_ids[node_id]

        executor.execute_flush(
            node_id=node_id,
            transform=MockBatchTransform(),
            ctx=PluginContext(),
            step_in_pipeline=0,
            trigger_type=TriggerType.END_OF_SOURCE,
        )

        batch = recorder.get_batch(batch_id)
        assert batch.trigger_reason == "end_of_source"
```

### Step 9: Add integration test for recovery

**File:** `tests/engine/test_orchestrator_recovery.py`

```python
def test_recovery_detects_interrupted_batch_flush(self, ...):
    """Interrupted batch flush should be detectable for recovery."""
    # 1. Start a run with aggregation
    # 2. Buffer rows into aggregation
    # 3. Transition batch to "executing" (simulating crash during flush)
    # 4. Call recovery
    # 5. Verify recovery finds the executing batch and marks it failed
```

## Testing Checklist

- [ ] `execute_flush()` creates a node_state record for the batch operation
- [ ] `execute_flush()` transitions batch: draft → executing → completed
- [ ] Failed transform transitions batch to "failed"
- [ ] Exception during flush transitions batch to "failed" and re-raises
- [ ] `trigger_reason` is recorded (COUNT, TIMEOUT, CONDITION, END_OF_SOURCE)
- [ ] `aggregation_state_id` links batch to its node_state
- [ ] Recovery can detect batches stuck in "executing" status
- [ ] End-of-source flush uses TriggerType.END_OF_SOURCE
- [ ] Input hash computed from all buffered rows
- [ ] Output hash computed from transform result

## Run Tests

```bash
# Run new tests
.venv/bin/python -m pytest tests/engine/test_aggregation_audit.py -v

# Run existing aggregation tests
.venv/bin/python -m pytest tests/engine/ -k aggregation -v

# Run recovery tests
.venv/bin/python -m pytest tests/engine/test_orchestrator_recovery.py -v

# Run full engine test suite
.venv/bin/python -m pytest tests/engine/ -v
```

## Acceptance Criteria

1. ✅ Each aggregation flush creates a node_state at the aggregation node
2. ✅ Node state includes input hash (all buffered rows) and output hash
3. ✅ Batches transition through expected statuses: draft → executing → completed/failed
4. ✅ Trigger reason recorded in batch (COUNT, TIMEOUT, END_OF_SOURCE, etc.)
5. ✅ Batch links to its flush node_state via `aggregation_state_id`
6. ✅ Recovery can detect and handle interrupted flushes
7. ✅ Exception during flush records failure and re-raises

## Security Notes

**Why this matters for audit integrity:**
- Aggregation is a transform boundary - CLAUDE.md requires "Input AND output captured at every transform"
- Without node_state, there's no proof of what the aggregation received or produced
- Batch status tracking enables crash recovery without reprocessing already-flushed batches

## Dependencies

This fix has no external dependencies. It uses existing:
- `LandscapeRecorder.begin_node_state()` / `complete_node_state()`
- `LandscapeRecorder.update_batch_status()` / `complete_batch()`
- `TriggerType` enum
- `stable_hash()` for input/output hashing

---

## Implementation Summary

**Status:** Completed
**Commits:** See git history for this feature
**Notes:** Aggregation batch audit trail fixed to properly record node_state for flush operations and transition batch status through expected lifecycle states.
