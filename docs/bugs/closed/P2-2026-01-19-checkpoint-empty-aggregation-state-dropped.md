# Bug Report: CheckpointManager drops empty aggregation_state due to truthiness check

## Summary

- `CheckpointManager.create_checkpoint()` serializes `aggregation_state` with `if aggregation_state else None`.
- This treats an empty dict (`{}`) as “no state” and stores `NULL`/`None` instead of `"{}"`.
- If/when aggregation checkpointing is used, this can break resume determinism and makes it impossible to distinguish “empty state” from “state omitted”.

## Severity

- Severity: minor
- Priority: P3

## Reporter

- Name or handle: codex
- Date: 2026-01-19
- Related run/issue ID: N/A

## Environment

- Commit/branch: `8cfebea78be241825dd7487fed3773d89f2d7079` (local)
- OS: Linux (Ubuntu kernel 6.8.0-90-generic)
- Python version: 3.13.1
- Config profile / env vars: N/A
- Data set or fixture: N/A

## Agent Context (if relevant)

- Goal or task prompt: deep dive into subsystem 3 (core infrastructure) and create bug tickets
- Model/version: GPT-5.2 (Codex CLI)
- Tooling and permissions (sandbox/approvals): workspace-write, network restricted, approvals on-request
- Determinism details (seed, run ID): N/A
- Notable tool calls or steps: static inspection of `src/elspeth/core/checkpoint/manager.py`

## Steps To Reproduce

1. Call `CheckpointManager.create_checkpoint(..., aggregation_state={})`.
2. Inspect the returned `Checkpoint.aggregation_state_json` (or the stored DB row).

## Expected Behavior

- Empty state should serialize to `"{}"` (or an explicit empty representation), preserving the difference between “no aggregation state provided” and “aggregation state is empty”.

## Actual Behavior

- Empty dict evaluates falsey, so `aggregation_state_json` becomes `None`.

## Evidence

- Truthiness check drops empty dict:
  - `src/elspeth/core/checkpoint/manager.py:57` (`json.dumps(aggregation_state) if aggregation_state else None`)

## Impact

- User-facing impact: resume behavior can diverge depending on whether state was empty vs absent.
- Data integrity / security impact: low-to-moderate (reproducibility correctness).
- Performance or cost impact: N/A.

## Root Cause Hypothesis

- A convenience truthiness check was used instead of an explicit `is not None` check.

## Proposed Fix

- Code changes (modules/files):
  - `src/elspeth/core/checkpoint/manager.py`:
    - Change to `json.dumps(aggregation_state) if aggregation_state is not None else None`
- Tests to add/update:
  - Add a unit test asserting `{}` is stored as `"{}"` and `None` is stored as `NULL`.
- Risks or migration steps:
  - None; this only changes serialization of empty dict.

## Architectural Deviations

- Spec or doc reference: N/A (checkpoint correctness)
- Observed divergence: empty state is indistinguishable from absent state.
- Reason (if known): truthiness check.
- Alignment plan or decision needed: none.

## Acceptance Criteria

- Empty aggregation state persists and round-trips distinctly from `None`.

## Tests

- Suggested tests to run: `.venv/bin/python -m pytest tests/integration/test_checkpoint_recovery.py`
- New tests required: yes (unit coverage in checkpoint manager tests)

## Notes / Links

- Related issues/PRs: N/A

## Resolution

**Status:** CLOSED (2026-01-21)
**Resolved by:** Claude Opus 4.5

### Timeline

1. **Bug discovered:** 2026-01-19 via static analysis at commit `8cfebea`
2. **Code fixed:** Prior to bug report filing (commit `07084c3` - "chore: delint and reformat codebase with line-length 140")
3. **Tests added:** 2026-01-21 - regression tests added to `tests/core/checkpoint/test_manager.py`

### Changes Made

**Code (already fixed before report):**
- `src/elspeth/core/checkpoint/manager.py:57`: Changed from `if aggregation_state else None` to `if aggregation_state is not None else None`

**Tests added:**
- `test_checkpoint_with_empty_aggregation_state_preserved()`: Verifies `{}` serializes to `"{}"`
- `test_checkpoint_with_none_aggregation_state_is_null()`: Verifies `None` stays as `None`

### Verification

```bash
.venv/bin/python -m pytest tests/core/checkpoint/test_manager.py -v
# 9 tests passed including both new regression tests
```

### Notes

The fix was applied in the delint commit before the bug report was filed, but no regression tests existed. The resolution adds proper test coverage to prevent future regressions of this common Python truthiness bug.
