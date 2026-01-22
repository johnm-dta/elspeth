# Bug Report: RetryManager calls on_retry even when no retry will occur

## Summary

- `RetryManager.execute_with_retry()` invokes `on_retry` for any retryable exception, even on the final attempt when retries are exhausted, which can record phantom “retry scheduled” events.

## Severity

- Severity: minor
- Priority: P3

## Reporter

- Name or handle: Codex
- Date: 2026-01-21
- Related run/issue ID: N/A

## Environment

- Commit/branch: ae2c0e6f088f467276582fa8016f91b4d3bb26c7 (fix/rc1-bug-burndown-session-2)
- OS: Linux nyx.foundryside.dev 6.8.0-90-generic
- Python version: 3.13.1
- Config profile / env vars: N/A
- Data set or fixture: N/A

## Agent Context (if relevant)

- Goal or task prompt: Deep dive into src/elspeth/engine/retry.py for bugs.
- Model/version: Codex (GPT-5)
- Tooling and permissions (sandbox/approvals): Codex CLI, workspace-write sandbox
- Determinism details (seed, run ID): N/A
- Notable tool calls or steps: Manual code inspection only

## Steps To Reproduce

1. Create `RetryManager(RetryConfig(max_attempts=1))`.
2. Call `execute_with_retry()` with a retryable exception and an `on_retry` callback.
3. Observe `on_retry` fires even though no further attempt will happen.

## Expected Behavior

- `on_retry` should be called only when a subsequent retry is actually scheduled.

## Actual Behavior

- `on_retry` is called for every retryable failure, including the final attempt before `MaxRetriesExceeded`.

## Evidence

- `src/elspeth/engine/retry.py`: `on_retry` is gated only on `is_retryable(e)` and does not check remaining attempts.
- Comment in code says “Only call on_retry for retryable errors that will be retried.”

## Impact

- User-facing impact: Audit/logging hooks can record retries that never occur.
- Data integrity / security impact: Audit trail can become misleading about attempted recovery.
- Performance or cost impact: None.

## Root Cause Hypothesis

- `on_retry` is invoked before tenacity evaluates stop conditions, without checking if another attempt will run.

## Proposed Fix

- Code changes (modules/files):
  - Move `on_retry` into a tenacity `before_sleep` hook or check `attempt < max_attempts` before calling.
- Config or schema changes: None.
- Tests to add/update:
  - Add test asserting `on_retry` is not called when no retries remain.
- Risks or migration steps: None.

## Architectural Deviations

- Spec or doc reference: inline comment in `src/elspeth/engine/retry.py`.
- Observed divergence: Callback fires even when no retry will happen.
- Reason (if known): Callback is triggered without consulting stop condition.
- Alignment plan or decision needed: Define semantics of `on_retry` and enforce them.

## Acceptance Criteria

- `on_retry` fires only when a new retry attempt will execute.

## Tests

- Suggested tests to run: `pytest tests/engine/test_retry.py -k records_attempts`
- New tests required: Yes.

## Notes / Links

- Related issues/PRs: N/A
- Related design docs: N/A
