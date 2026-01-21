# Bug Report: RetryManager on_retry attempt numbers are 1-based, while audit uses 0-based

## Summary

- `RetryManager.execute_with_retry()` passes Tenacity's 1-based `attempt_number` to `on_retry`, but engine audit attempt numbering is 0-based (first attempt = 0), so callback consumers will record misaligned attempt indices.

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

1. Use `RetryManager.execute_with_retry()` with an `on_retry` callback to record attempt numbers.
2. Compare recorded attempt numbers to node state attempts (`attempt=0` for first attempt).
3. Observe mismatch: on_retry reports 1 for the first failure, audit uses 0 for first attempt.

## Expected Behavior

- Retry callbacks should use the same attempt indexing as audit records (0-based), or explicitly document a different convention.

## Actual Behavior

- `on_retry` receives Tenacity's 1-based attempt numbers.

## Evidence

- `src/elspeth/engine/retry.py` sets `attempt = attempt_state.retry_state.attempt_number` and passes it to `on_retry`.
- Audit recorder docs: `src/elspeth/core/landscape/recorder.py` notes “attempt number (0 for first attempt)”.

## Impact

- User-facing impact: Retry audit hooks can produce off-by-one attempt indices.
- Data integrity / security impact: Potential mismatch or conflicts if attempt numbers are used in unique keys or lineage.
- Performance or cost impact: None.

## Root Cause Hypothesis

- RetryManager uses tenacity's attempt numbering without normalizing to engine conventions.

## Proposed Fix

- Code changes (modules/files):
  - Normalize attempt number before calling `on_retry` (e.g., `attempt_number - 1`).
  - Or document that `on_retry` is 1-based and update consumers accordingly.
- Config or schema changes: None.
- Tests to add/update:
  - Add test asserting callback attempt numbering matches audit convention.
- Risks or migration steps: If changing numbering, update any existing consumers.

## Architectural Deviations

- Spec or doc reference: `src/elspeth/core/landscape/recorder.py` (0-based attempts).
- Observed divergence: Retry callback uses 1-based attempts.
- Reason (if known): Direct passthrough from tenacity.
- Alignment plan or decision needed: Standardize attempt indexing.

## Acceptance Criteria

- Retry callback attempt numbers align with audit attempt indexing.

## Tests

- Suggested tests to run: `pytest tests/engine/test_retry.py -k records_attempts`
- New tests required: Yes.

## Notes / Links

- Related issues/PRs: N/A
- Related design docs: N/A
