# Bug Report: Checkpoint ID truncation risks collisions under high volume

## Summary

- Checkpoint ID uses `uuid.uuid4().hex[:12]` (48 bits of entropy). Birthday paradox gives 1% collision probability at ~2.6 million checkpoints.

## Severity

- Severity: trivial
- Priority: P3

## Reporter

- Name or handle: Codex (triaged by Claude)
- Date: 2026-01-31

## Evidence

- `src/elspeth/core/checkpoint/manager.py:78` - `f"cp-{uuid.uuid4().hex[:12]}"`
- 48 bits = 1% collision at 2.6M checkpoints

## Proposed Fix

- Use full UUID hex (32 chars) for collision-free IDs

## Acceptance Criteria

- Checkpoint IDs use full UUID

## Verification (2026-02-01)

**Status: STILL VALID**

- Checkpoint IDs are still truncated to 12 hex chars (`uuid.uuid4().hex[:12]`). (`src/elspeth/core/checkpoint/manager.py:78`)

## Closure Report (2026-02-01)

**Status:** CLOSED (IMPLEMENTED)

### Fix Summary

- Use full UUID hex for checkpoint IDs to eliminate collision risk.

### Test Coverage

- Not added (trivial ID length change).
