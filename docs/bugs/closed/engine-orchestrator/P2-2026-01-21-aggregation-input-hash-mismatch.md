# Bug Report: Aggregation flush uses inconsistent input hashes (node_state vs result)

## Summary

- `AggregationExecutor.execute_flush()` computes `input_hash` from `buffered_rows`, but `begin_node_state()` computes the stored input hash from a wrapped dict (`{"batch_rows": buffered_rows}`).
- The result's `input_hash` therefore does not match the node_state input hash, breaking audit consistency and traceability for aggregation inputs.

## Severity

- Severity: major
- Priority: P2

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
- Notable tool calls or steps: code inspection of `src/elspeth/engine/executors.py` and recorder hashing logic

## Steps To Reproduce

1. Run an aggregation flush and inspect the recorded `node_states.input_hash` in Landscape.
2. Compare it to `TransformResult.input_hash` returned by `execute_flush()` for the same flush.

## Expected Behavior

- The node_state input hash and the result's input_hash match, since they describe the same input payload.

## Actual Behavior

- The node_state input hash is computed from `{"batch_rows": buffered_rows}` while `result.input_hash` is computed from `buffered_rows`, so the hashes differ.

## Evidence

- `input_hash` computed from list of rows:
  - `src/elspeth/engine/executors.py:893`
  - `src/elspeth/engine/executors.py:964`
- Node state input hash computed from wrapped dict (via `begin_node_state` hashing input_data):
  - `src/elspeth/engine/executors.py:907`
  - `src/elspeth/engine/executors.py:909`
  - `src/elspeth/core/landscape/recorder.py:1019`

## Impact

- User-facing impact: audit UI and exports show inconsistent input hashes for the same aggregation flush.
- Data integrity / security impact: breaks hash-based verification and traceability guarantees for aggregation inputs.
- Performance or cost impact: none direct.

## Root Cause Hypothesis

- Aggregation flush uses two different representations of the input when hashing.

## Proposed Fix

- Code changes (modules/files):
  - `src/elspeth/engine/executors.py`: either compute `input_hash` from the same `batch_input` dict passed to `begin_node_state`, or pass `buffered_rows` directly as `input_data` so both hashes align.
- Config or schema changes: none.
- Tests to add/update:
  - Add a test asserting `result.input_hash == node_state.input_hash` for aggregation flushes.
- Risks or migration steps: none.

## Architectural Deviations

- Spec or doc reference (e.g., docs/design/architecture.md#L...): auditability standard in `CLAUDE.md` (transform boundaries should be traceable by hash).
- Observed divergence: mismatch between state and result hashes.
- Reason (if known): input wrapper for node_state differs from result hash input.
- Alignment plan or decision needed: standardize hash input representation for aggregation flushes.

## Acceptance Criteria

- Aggregation flush results and node_states share identical input_hash values.

## Tests

- Suggested tests to run:
  - `pytest tests/engine/test_processor.py -k aggregation_input_hash`
- New tests required: yes (input hash consistency).

## Notes / Links

- Related issues/PRs: N/A
- Related design docs: N/A

---

## VERIFICATION: 2026-01-25

**Status:** STILL VALID

**Verified By:** Claude Code P2 verification wave 3

**Current Code Analysis:**

Examined current code in `/home/john/elspeth-rapid/src/elspeth/engine/executors.py` on branch `fix/rc1-bug-burndown-session-4`:

**Line 905:** `input_hash = stable_hash(buffered_rows)`
- Computes hash from the raw list of buffered rows

**Lines 919-926:**
```python
batch_input: dict[str, Any] = {"batch_rows": buffered_rows}
state = self._recorder.begin_node_state(
    ...
    input_data=batch_input,
    ...
)
```
- Wraps `buffered_rows` in a dict with key `"batch_rows"`
- Passes this wrapped dict to `begin_node_state()`

**Line 993:** `result.input_hash = input_hash`
- Assigns the hash of the unwrapped list to the result

In `recorder.py` line 1045, `begin_node_state()` computes: `input_hash = stable_hash(input_data)`

This creates a hash mismatch:
- `result.input_hash` = `stable_hash(buffered_rows)` (list)
- `state.input_hash` (in DB) = `stable_hash({"batch_rows": buffered_rows})` (wrapped dict)

These will be different hash values for the same logical input.

**Git History:**

Checked commits since 2026-01-21:
- `54edba7` (2026-01-24): Added defensive guard for buffer/token length mismatch - unrelated
- `57c57f5` (2026-01-21): Fixed 8 RC1 bugs - did not address this hash mismatch issue
- No commits found that modify the aggregation input hash computation logic

Searched for related commits with:
- `git log --all --oneline --grep="aggregation.*hash|input.*hash.*aggregation|batch.*input.*hash"` - no results

**Root Cause Confirmed:**

YES - The bug is still present. The code still exhibits the exact behavior described in the original report:

1. `execute_flush()` computes `input_hash` from `buffered_rows` (line 905)
2. `begin_node_state()` receives `{"batch_rows": buffered_rows}` and computes hash from the wrapped dict (line 924)
3. Result and node_state have different input hashes for the same logical input

**Impact Assessment:**

- **Audit integrity violation**: The audit trail records one hash in `node_states.input_hash` while `TransformResult` carries a different hash for the same input data
- **Traceability broken**: Cannot reliably verify aggregation inputs by comparing hashes
- **No test coverage**: Examined `tests/engine/test_executors.py` - aggregation flush tests (`test_checkpoint_restore_then_flush_succeeds`, `test_execute_flush_detects_incomplete_restoration`) verify flush succeeds but do NOT assert `result.input_hash == state.input_hash`

**Recommendation:**

**Keep open** - This is a valid P2 audit integrity bug that violates ELSPETH's auditability standard. The fix is straightforward (align the hash input representation), low-risk, and should include a regression test to verify `result.input_hash == node_state.input_hash` after aggregation flush.

---

## RESOLUTION: 2026-01-28

**Status:** FIXED

**Fixed By:** Claude Code

**Branch:** `feat/structured-outputs`

**Fix Summary:**

Moved `input_hash` computation from BEFORE the wrapping step to AFTER the wrapping step in `AggregationExecutor.execute_flush()`.

**Code Changes:**

1. **`src/elspeth/engine/executors.py`** (~line 1054-1058):
   - Removed early `input_hash = stable_hash(buffered_rows)` computation
   - Added `input_hash = stable_hash(batch_input)` AFTER creating `batch_input = {"batch_rows": buffered_rows}`
   - Added comment referencing this bug report

2. **`tests/engine/test_aggregation_audit.py`**:
   - Added regression test `test_flush_result_hash_matches_node_state_hash` that explicitly asserts `result.input_hash == agg_state.input_hash`

**Verification:**

- All 10 aggregation audit tests pass
- New regression test verifies hash consistency
- No type errors (mypy clean)

**Acceptance Criteria Met:**

- [x] `result.input_hash == node_state.input_hash` for all aggregation flushes
- [x] New regression test verifies hash consistency
- [x] All existing aggregation tests pass
- [x] No type errors introduced
