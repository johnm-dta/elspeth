# Bug Report: Quarantined outcomes recorded before sink durability

## Summary

- `record_token_outcome(QUARANTINED)` is called immediately when a row fails, before the quarantine sink actually writes the data. If the sink write fails, the QUARANTINED outcome is recorded but no durable output exists.

## Severity

- Severity: major
- Priority: P1

## Reporter

- Name or handle: Codex (triaged by Claude)
- Date: 2026-01-31
- Related run/issue ID: N/A

## Evidence

- `src/elspeth/engine/orchestrator.py:1194-1229` - `record_token_outcome(QUARANTINED)` is called immediately after creating the quarantine token.
- `src/elspeth/engine/orchestrator.py:1231-1232` - token is enqueued into `pending_tokens[quarantine_sink]` for later sink write.
- `src/elspeth/engine/orchestrator.py:1586-1623` - sink writes occur later, outside the source loop, via `SinkExecutor.write()`.
- If the quarantine sink write fails, the QUARANTINED outcome has already been recorded without durable output.

## Impact

- User-facing impact: Audit trail shows QUARANTINED but data may not exist in sink
- Data integrity / security impact: Terminal outcome without durability violates sink contract
- Performance or cost impact: None

## Root Cause Hypothesis

- Outcome recording happens before sink durability is confirmed, violating the "outcome = durable output" invariant.

## Proposed Fix

- Code changes:
  - Record QUARANTINED outcome AFTER sink write succeeds
  - Or: Record outcome with "pending" status, confirm after sink durability
- Tests to add/update:
  - Add test that mocks sink.write() to fail for quarantine, verify no QUARANTINED outcome

## Acceptance Criteria

- QUARANTINED outcome is only recorded after sink confirms durability
- Failed sink writes result in appropriate error handling, not false outcomes

## Resolution (2026-02-01)

**Status: FIXED**

### Fix Summary

Introduced `PendingOutcome` dataclass to carry outcome information through the `pending_tokens` queue, deferring outcome recording until after sink durability is confirmed.

### Changes Made

1. **New dataclass `PendingOutcome`** (`contracts/engine.py`):
   - Holds `outcome: RowOutcome` and optional `error_hash: str | None`
   - Used instead of raw `RowOutcome` in `pending_tokens`
   - Carries error_hash for QUARANTINED outcomes through to sink executor
   - Placed in contracts module as a shared type (not in orchestrator to avoid circular imports)

2. **Updated `pending_tokens` type** (multiple locations in orchestrator.py):
   - Changed from `dict[str, list[tuple[TokenInfo, RowOutcome | None]]]`
   - To `dict[str, list[tuple[TokenInfo, PendingOutcome | None]]]`

3. **Fixed quarantine flow** (`orchestrator.py`):
   - Removed premature `recorder.record_token_outcome()` call
   - Now passes `PendingOutcome(RowOutcome.QUARANTINED, error_hash)` to pending_tokens
   - Outcome recorded by SinkExecutor.write() AFTER sink durability

4. **Updated `SinkExecutor.write()`** (`executors.py`):
   - Changed `outcome` parameter to `pending_outcome: PendingOutcome | None`
   - Now passes `error_hash` when recording outcomes
   - All outcome types (COMPLETED, ROUTED, QUARANTINED) use same post-durability path

5. **Updated groupby logic** in sink write loops:
   - Sort key function updated for `PendingOutcome` structure
   - Groups by (outcome, error_hash) for correct batching

### Test Added

`test_quarantine_outcome_not_recorded_if_sink_fails` (`test_orchestrator_errors.py`):
- Verifies that when quarantine sink fails, NO QUARANTINED outcome is recorded
- Confirms the "outcome = durable output" invariant is maintained

### Files Changed

- `src/elspeth/contracts/engine.py` - Added PendingOutcome dataclass
- `src/elspeth/contracts/__init__.py` - Export PendingOutcome
- `src/elspeth/engine/orchestrator.py` - Import from contracts, updated all pending_tokens usage
- `src/elspeth/engine/executors.py` - Import from contracts, updated SinkExecutor.write() signature
- `tests/engine/test_orchestrator_errors.py` - Added failing-sink test
