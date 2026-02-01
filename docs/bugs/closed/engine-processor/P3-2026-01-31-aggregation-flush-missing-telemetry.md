# Bug Report: Aggregation Flushes Do Not Emit TransformCompleted Telemetry

## Summary

- Batch-aware aggregation flushes never emit `TransformCompleted` telemetry events, creating a gap in row-level telemetry.

## Severity

- Severity: minor
- Priority: P3

## Reporter

- Name or handle: Codex (triaged by Claude)
- Date: 2026-01-31

## Evidence

- `src/elspeth/engine/processor.py:480-519` - `execute_flush()` called with no telemetry emission
- Regular transforms DO call `_emit_transform_completed()` (lines 1684-1698)
- Audit trail (Landscape) is still correct; only telemetry gap

## Proposed Fix

- Call `_emit_transform_completed()` after `execute_flush()` returns

## Acceptance Criteria

- Aggregation flushes emit TransformCompleted telemetry events

## Verification (2026-02-01)

**Status: STILL VALID**

- Aggregation flush path still calls `execute_flush()` without emitting `TransformCompleted`. (`src/elspeth/engine/processor.py:485-491`)
- Regular transforms still emit `TransformCompleted` for comparison. (`src/elspeth/engine/processor.py:1609-1615`)

## Resolution (2026-02-02)

**Status: FIXED**

Added `_emit_transform_completed()` calls after successful `execute_flush()` in both aggregation flush paths:

1. **`handle_timeout_flush`** (timeout/end-of-source flushes):
   - Added telemetry emission at `processor.py:552-560` (after failure path returns)
   - Emits one `TransformCompleted` event per buffered token

2. **`_process_batch_aggregation_node`** (count-triggered flushes):
   - Added telemetry emission at `processor.py:833-841` (after failure path returns)
   - Emits one `TransformCompleted` event per buffered token

**Design decision:** Emit per-token rather than per-batch for consistency with regular transform telemetry. This allows:
- Accurate token counting in observability dashboards
- Consistent aggregation patterns in telemetry queries
- Each buffered token gets credited with the batch processing time

**Tests added:**
- `TestAggregationFlushTelemetry::test_aggregation_count_flush_emits_transform_completed`
- `TestAggregationFlushTelemetry::test_aggregation_end_of_source_flush_emits_transform_completed`

All 12 telemetry tests pass, plus 796 engine/integration tests.
