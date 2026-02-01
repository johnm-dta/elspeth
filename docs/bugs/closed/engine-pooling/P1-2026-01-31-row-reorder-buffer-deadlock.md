# Bug Report: RowReorderBuffer deadlocks when evicting a non-head sequence

## Summary

- RowReorderBuffer can deadlock when a sequence that is not at the head of the queue is evicted (e.g., due to retry/timeout), because the gap-skipping logic only runs at eviction time, not at release time.

## Severity

- Severity: major
- Priority: P1

## Reporter

- Name or handle: Codex (triaged by Claude)
- Date: 2026-01-31
- Related run/issue ID: N/A

## Evidence

- `src/elspeth/plugins/batching/row_reorder_buffer.py:248-289` - `wait_for_next_release()` only releases when `_next_release_seq` exists in `_pending`
- Lines 321-323 - `evict()` only advances if evicting the current head
- Gap-skipping in `evict()` only runs at eviction time, not at release time
- If seq 2 is evicted while seq 0 is still head, and later seq 0 and 1 release, the release loop blocks at seq 2 forever

## Impact

- User-facing impact: Pipeline hangs indefinitely during retry/timeout scenarios
- Data integrity / security impact: No data corruption, but run never completes
- Performance or cost impact: Complete pipeline stall

## Root Cause Hypothesis

- The gap-skipping logic should run in `wait_for_next_release()` when the expected sequence is not in `_pending` but is recorded as evicted.

## Proposed Fix

- Code changes:
  - Add gap-skipping logic to `wait_for_next_release()` to detect and skip evicted sequences
  - Or: maintain an evicted set and check it before blocking
- Tests to add/update:
  - Add test that evicts a non-head sequence, then releases head, verify no deadlock

## Acceptance Criteria

- Evicting a non-head sequence does not cause deadlock
- Release loop properly skips evicted sequences
