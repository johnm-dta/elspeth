# Bug Report: Passthrough aggregation failure leaves buffered tokens non-terminal

## Summary

- When a passthrough aggregation flush returns an error, only the triggering token is marked FAILED. Previously buffered tokens remain in the BUFFERED outcome even though the batch is cleared, so they never reach a terminal state.

## Severity

- Severity: major
- Priority: P1

## Reporter

- Name or handle: Codex
- Date: 2026-01-21
- Related run/issue ID: N/A

## Environment

- Commit/branch: ae2c0e6 / fix/rc1-bug-burndown-session-2
- OS: Linux
- Python version: Python 3.13.1
- Config profile / env vars: N/A
- Data set or fixture: N/A

## Agent Context (if relevant)

- Goal or task prompt: Deep dive src/elspeth/engine/processor.py for bugs; create reports.
- Model/version: GPT-5 (Codex CLI)
- Tooling and permissions (sandbox/approvals): workspace-write, no escalations
- Determinism details (seed, run ID): N/A
- Notable tool calls or steps: Code inspection only

## Steps To Reproduce

1. Configure a batch-aware transform with aggregation `output_mode: passthrough`.
2. Make the batch transform return `TransformResult.error(...)` on flush.
3. Feed multiple rows so at least one row is buffered before the flush.

## Expected Behavior

- All tokens in the failed batch transition to a terminal outcome (FAILED/QUARANTINED) or are retried explicitly.

## Actual Behavior

- Only the triggering token is marked FAILED; earlier buffered tokens stay at BUFFERED even though the batch buffer is cleared.

## Evidence

- `src/elspeth/engine/processor.py:202-222` records FAILED for only `current_token` on flush error.
- Passthrough buffering records BUFFERED outcomes for prior tokens (`src/elspeth/engine/processor.py:385-401`).
- `AggregationExecutor.execute_flush()` clears buffers on failure (`src/elspeth/engine/executors.py:1006-1017`), so buffered tokens will never be reprocessed.

## Impact

- User-facing impact: Rows disappear silently on batch failure.
- Data integrity / security impact: Violates “every token reaches terminal state”; audit trail shows stuck BUFFERED outcomes.
- Performance or cost impact: Manual remediation required; batch retry logic undermined.

## Root Cause Hypothesis

- Error handling in `_process_batch_aggregation_node()` only records failure for the triggering token and ignores buffered tokens.

## Proposed Fix

- Code changes (modules/files): `src/elspeth/engine/processor.py`
- Config or schema changes: None
- Tests to add/update: Add a test ensuring all buffered tokens receive terminal outcomes on flush error.
- Risks or migration steps: None; ensures audit completeness.

## Architectural Deviations

- Spec or doc reference (e.g., docs/design/architecture.md#L...): CLAUDE.md “Every row reaches exactly one terminal state”.
- Observed divergence: Buffered tokens can be left non-terminal after a failed flush.
- Reason (if known): Flush error path only handles `current_token`.
- Alignment plan or decision needed: Define terminal outcome for failed batches (likely FAILED/QUARANTINED for all members).

## Acceptance Criteria

- All tokens in a failed passthrough batch receive terminal outcomes.
- No BUFFERED outcomes remain after a failed flush.

## Tests

- Suggested tests to run: `pytest tests/engine/test_processor.py -k aggregation_passthrough_failure`
- New tests required: Yes (batch failure terminal outcomes).

## Notes / Links

- Related issues/PRs: N/A
- Related design docs: `docs/design/subsystems/00-overview.md`

## Verification (2026-01-25)

**Status: STILL VALID**

### Code Analysis

Examined current codebase at commit 7540e57 on branch `fix/rc1-bug-burndown-session-4`.

The bug remains unfixed. The error handling path in `src/elspeth/engine/processor.py:204-224` only records FAILED outcome for `current_token` (the triggering token) and immediately returns:

```python
if result.status != "success":
    error_msg = "Batch transform failed"
    error_hash = hashlib.sha256(error_msg.encode()).hexdigest()[:16]
    self._recorder.record_token_outcome(
        run_id=self._run_id,
        token_id=current_token.token_id,  # ← Only triggering token
        outcome=RowOutcome.FAILED,
        error_hash=error_hash,
    )
    return (  # ← Exits immediately, buffered_tokens ignored
        RowResult(...),
        child_items,
    )
```

The `buffered_tokens` variable (returned by `execute_flush()` on line 196) contains all previously buffered tokens that had BUFFERED outcomes recorded (lines 376-393). These tokens are never processed when flush fails.

### Evidence from executors.py

`AggregationExecutor.execute_flush()` at line 1046-1049 unconditionally clears buffers regardless of success/failure:

```python
# Step 6: Reset for next batch and clear buffers
self._reset_batch_state(node_id)
self._buffers[node_id] = []
self._buffer_tokens[node_id] = []  # ← Cleared on ALL paths
```

This means buffered tokens cannot be reprocessed - they are permanently lost.

### Test Coverage Gap

Searched test suite for coverage of this scenario:
- `tests/engine/test_aggregation_audit.py` has tests for flush failures (`test_failed_flush_marks_batch_failed`, `test_error_result_marks_batch_failed`)
- However, these tests only verify batch status and node_state, NOT token outcomes for buffered tokens
- No tests verify passthrough mode specifically with flush failures
- Test file `tests/engine/test_processor_outcomes.py` (added in commit 73bb99f for AUD-001) does not cover this edge case

### Git History

Checked commits since 2026-01-21:
- Commit c6afc31 "fix(processor): add missing outcome recordings for batch aggregation (AUD-001)" added outcome recordings for SUCCESS paths only
- Commit 73bb99f "test: add comprehensive outcome recording tests (AUD-001)" added outcome tests but did not cover flush failure + buffered tokens
- No commits address the flush failure path for buffered tokens

### Severity Confirmation

This bug violates the core architectural principle from CLAUDE.md: "Every row reaches exactly one terminal state - no silent drops."

**Impact:**
- Data loss: Buffered rows silently disappear from pipeline
- Audit trail corruption: BUFFERED outcomes with no subsequent terminal outcome
- No recovery path: Buffers cleared, tokens cannot be retried

### Line Number Updates

Bug report line references remain accurate:
- Error handling: lines 204-224 (currently accurate)
- Buffering path: lines 376-393 (report said 385-401, now 376-393 due to code growth)
- Buffer clearing: executors.py:1046-1049 (report said 1006-1017, now 1046-1049)

### Reproducibility

Bug is reproducible by:
1. Creating passthrough aggregation with count trigger = 3
2. Buffering 2 rows (both get BUFFERED outcome)
3. 3rd row triggers flush that returns `TransformResult.error(...)`
4. Only row 3 gets FAILED outcome
5. Rows 1 and 2 stuck with BUFFERED outcome forever

**Recommendation: This bug should be prioritized for immediate fix as it causes silent data loss and violates audit integrity guarantees.**
