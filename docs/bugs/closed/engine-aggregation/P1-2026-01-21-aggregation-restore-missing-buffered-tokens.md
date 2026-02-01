# Bug Report: Aggregation checkpoint restore drops buffered tokens, flush crashes on resume

## Summary

- `AggregationExecutor.restore_from_checkpoint()` restores buffered rows but clears `_buffer_tokens` and never reconstructs `TokenInfo`.
- `execute_flush()` assumes `_buffer_tokens` is populated and unconditionally uses `buffered_tokens[0]`, so a resumed aggregation flush can crash with `IndexError` or lose token metadata needed for audit lineage.

## Severity

- Severity: major
- Priority: P1

## Reporter

- Name or handle: codex
- Date: 2026-01-21
- Related run/issue ID: N/A

## Environment

- Commit/branch: `ae2c0e6f088f467276582fa8016f91b4d3bb26c7` (main)
- OS: Linux 6.8.0-90-generic
- Python version: 3.13.1
- Config profile / env vars: N/A
- Data set or fixture: N/A

## Agent Context (if relevant)

- Goal or task prompt: deep dive into `src/elspeth/engine/executors.py` for bugs and create reports
- Model/version: GPT-5 (Codex CLI)
- Tooling and permissions (sandbox/approvals): workspace-write, network restricted, approvals on-request
- Determinism details (seed, run ID): N/A
- Notable tool calls or steps: code inspection of `src/elspeth/engine/executors.py`

## Steps To Reproduce

1. Create an `AggregationExecutor` and buffer at least one row for an aggregation node.
2. Call `get_checkpoint_state()` to capture the buffer state.
3. Create a new `AggregationExecutor` and call `restore_from_checkpoint()` with the captured state.
4. Call `execute_flush()` for that node.

## Expected Behavior

- Restored buffers include the original `TokenInfo` (or are reconstructed), allowing `execute_flush()` to complete and preserve audit lineage.

## Actual Behavior

- `_buffer_tokens` is empty after restore; `execute_flush()` tries to read `buffered_tokens[0]` and crashes, or proceeds without token metadata.

## Evidence

- `execute_flush()` assumes `buffered_tokens[0]` exists:
  - `src/elspeth/engine/executors.py:886`
  - `src/elspeth/engine/executors.py:896`
  - `src/elspeth/engine/executors.py:897`
- `restore_from_checkpoint()` clears `_buffer_tokens` and never reconstructs `TokenInfo`:
  - `src/elspeth/engine/executors.py:1085`
  - `src/elspeth/engine/executors.py:1099`
  - `src/elspeth/engine/executors.py:1102`

## Impact

- User-facing impact: resume/flush can crash for aggregations, breaking recovery flows.
- Data integrity / security impact: missing token metadata breaks audit lineage (batch membership → node_state linkage).
- Performance or cost impact: failed resumes and repeated runs.

## Root Cause Hypothesis

- Checkpoint state stores token IDs but `restore_from_checkpoint()` does not rehydrate `TokenInfo`, leaving `_buffer_tokens` empty.

## Proposed Fix

- Code changes (modules/files):
  - `src/elspeth/engine/executors.py`: reconstruct `TokenInfo` from stored token IDs (query tokens table or accept token metadata in checkpoint), or store full token metadata in checkpoint state.
  - Guard `execute_flush()` against empty `_buffer_tokens` when `buffered_rows` exist (fail with a clear error and remediation).
- Config or schema changes: none.
- Tests to add/update:
  - Add a recovery test that buffers rows, checkpoints, restores, and flushes without error.
- Risks or migration steps: none.

## Architectural Deviations

- Spec or doc reference (e.g., docs/design/architecture.md#L...): `docs/design/requirements.md` (aggregation checkpoint/restore).
- Observed divergence: restored aggregation buffers are not executable.
- Reason (if known): restore path only restores rows, not token metadata.
- Alignment plan or decision needed: define how token metadata is persisted/restored for aggregation checkpoints.

## Acceptance Criteria

- Resumed aggregations can flush successfully with correct token lineage.
- `execute_flush()` no longer crashes on restored buffers.

## Tests

- Suggested tests to run:
  - `pytest tests/engine/test_processor.py -k aggregation_resume`
- New tests required: yes (aggregation checkpoint restore + flush).

## Notes / Links

- Related issues/PRs: N/A
- Related design docs: N/A

---

## Resolution

**Status**: FIXED
**Fixed in**: 260b9a7 through 91c95cd
**Release**: RC-2 (planned)

### Fix Summary

The root cause was that `get_checkpoint_state()` only stored `token_ids` (list of strings) instead of complete `TokenInfo` objects. On restoration, `restore_from_checkpoint()` attempted to query the database to reconstruct tokens, but:
1. The N+1 query pattern was slow
2. For checkpoints from previous runs, the database no longer had those token entries
3. This left `_buffer_tokens` empty while `_buffers` had rows, causing IndexError on flush

**Solution implemented:**
1. **Task 1** (260b9a7): Modified `get_checkpoint_state()` to store complete `TokenInfo` objects (token_id, row_id, branch_name, row_data) with 10MB size validation
2. **Task 2** (3e25073): Modified `restore_from_checkpoint()` to reconstruct `TokenInfo` from checkpoint data, eliminating database queries
3. **Task 3** (59bb35f, 54edba7): Added defensive guard in `execute_flush()` to crash explicitly if buffer/token length mismatch occurs
4. **Tasks 3.5-4** (30de5e6, 91c95cd): Added comprehensive tests for size validation and edge cases

**Benefits:**
- O(1) restoration (no database queries)
- Portable checkpoints (no database dependency)
- Clear error messages if corruption occurs
- Consistent with CoalesceExecutor pattern

### Testing

Added 16 comprehensive tests covering format migration, size validation, and edge cases:

**Checkpoint format tests:**
1. `test_get_checkpoint_state_stores_full_token_info` - Verifies checkpoint stores complete TokenInfo with all fields
2. `test_get_checkpoint_state_excludes_empty_buffers` - Empty buffers not included in checkpoint
3. `test_restore_from_checkpoint_reconstructs_full_token_info` - Full TokenInfo reconstruction from checkpoint data
4. `test_restore_from_checkpoint_restores_trigger_count` - Trigger evaluator count correctly advanced
5. `test_checkpoint_roundtrip` - Save/restore cycle preserves all state without data loss
6. `test_checkpoint_restore_then_flush_succeeds` - Flush works correctly after checkpoint restoration

**Defensive guard tests:**
7. `test_execute_flush_detects_incomplete_restoration` - Defensive guard catches buffer/token mismatch

**Size validation tests:**
8. `test_checkpoint_size_warning_at_1mb_threshold` - 1MB threshold logs warning with size info
9. `test_checkpoint_size_no_warning_under_1mb` - No warning logged for checkpoints under 1MB
10. `test_checkpoint_size_warning_but_no_error_between_thresholds` - 1-10MB range warns but doesn't error
11. `test_checkpoint_size_error_at_10mb_limit` - 10MB hard limit raises RuntimeError
12. `test_checkpoint_size_error_message_includes_solutions` - Error message includes actionable guidance

**Edge case and validation tests:**
13. `test_restore_from_checkpoint_handles_empty_state` - Empty checkpoint handled gracefully
14. `test_restore_from_checkpoint_crashes_on_missing_tokens_key` - Old format detected with clear error
15. `test_restore_from_checkpoint_crashes_on_invalid_tokens_type` - Invalid tokens type crashes with clear error
16. `test_restore_from_checkpoint_crashes_on_missing_token_fields` - Missing required fields crash with clear error

All tests pass including critical `test_checkpoint_roundtrip`.

### Breaking Changes

**Checkpoint format change:** Old checkpoints using `{"rows": [...], "token_ids": [...]}` format are no longer compatible. Restoration from old checkpoints will raise `ValueError` with clear migration instructions.

**Migration:** Existing checkpoints must be discarded and regenerated. This is acceptable for RC-1 to RC-2 as checkpoints are ephemeral state, not persistent artifacts.
