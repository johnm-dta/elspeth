# RC-2 Release Note: Aggregation Checkpoint Fix

## Summary

Fixed critical bug (P1-2026-01-21) where aggregation executors would crash with `IndexError` when flushing after checkpoint restoration.

## Root Cause

Checkpoint format only stored token IDs, requiring database queries to reconstruct full TokenInfo objects during restoration. This created:
1. Performance issues (N+1 queries)
2. Data loss when database no longer had token entries from previous runs
3. Buffer/token length mismatches causing IndexError on flush

## Fix

**Checkpoint format migration:**
- Old format: `{"rows": [...], "token_ids": [...], "batch_id": ...}`
- New format: `{"tokens": [{"token_id", "row_id", "branch_name", "row_data"}], "batch_id": ...}`

**Implementation:**
1. Store complete TokenInfo in checkpoints (10MB limit with 1MB warning)
2. Reconstruct TokenInfo from checkpoint data (no database queries)
3. Defensive guard in execute_flush() to detect corruption

**Commits:**
- 260b9a7: Store full TokenInfo with size validation (Task 1)
- 3e25073: Restore TokenInfo without DB queries (Task 2)
- 54edba7, 59bb35f: Add defensive guard for buffer/token mismatch (Task 3)
- 30de5e6, 91c95cd: Add size validation and edge case tests (Tasks 3.5-4)

## Impact

- Checkpoint restoration is now O(1) instead of O(N)
- Checkpoints are portable (no database dependency)
- Clear error messages on corruption
- **Breaking**: Old checkpoints incompatible, must regenerate

## Testing

16 new tests covering:
- Format migration and round-trip
- Size validation (1MB warning, 10MB error)
- Edge cases and validation
- Defensive guard behavior

All tests pass, including critical `test_checkpoint_roundtrip`.

## Migration Guide

**For users with existing checkpoints:**
1. Existing checkpoints from the prior format will fail restoration with clear error message
2. Solution: Discard old checkpoints and rerun pipelines
3. Checkpoints are ephemeral state, not persistent artifacts

**For developers:**
- Review `src/elspeth/engine/executors.py` lines 1088-1220 for new checkpoint format
- See tests in `tests/engine/test_executors.py` for usage examples
