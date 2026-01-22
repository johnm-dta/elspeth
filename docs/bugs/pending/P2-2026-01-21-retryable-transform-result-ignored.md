# Bug Report: Retryable TransformResult errors are never retried

## Summary

- RowProcessor only retries exceptions, not `TransformResult.error(retryable=True)`. This ignores the retryable flag specified in the plugin protocol and prevents transient errors from being retried when they are reported as error results instead of exceptions.

## Severity

- Severity: major
- Priority: P2

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

1. Configure retry policy in settings.
2. Implement a transform that catches a transient external error and returns `TransformResult.error(..., retryable=True)`.
3. Run the pipeline on data that triggers the transient error.

## Expected Behavior

- RetryManager should retry the transform according to policy when `retryable=True`.

## Actual Behavior

- No retry occurs; the error is routed or discarded immediately.

## Evidence

- RowProcessor explicitly states TransformResult errors are not retried and only exceptions trigger retries: `src/elspeth/engine/processor.py:436-437`.
- Retryable check only inspects exception types: `src/elspeth/engine/processor.py:472-475`.
- Plugin protocol requires retryable errors to be considered by engine policy: `docs/contracts/plugin-protocol.md:1366-1372`.

## Impact

- User-facing impact: Reduced resilience to transient API errors reported as result errors.
- Data integrity / security impact: Error routing outcomes differ from intended retry semantics.
- Performance or cost impact: Increased manual retries or failed runs.

## Root Cause Hypothesis

- Retry logic only wraps exceptions; result-level retryable flags are ignored.

## Proposed Fix

- Code changes (modules/files): `src/elspeth/engine/processor.py`, possibly `src/elspeth/engine/retry.py`
- Config or schema changes: None
- Tests to add/update: Add a retry test where TransformResult.error(retryable=True) is retried per policy.
- Risks or migration steps: Clarify whether retryable on results is supported or deprecated; update docs accordingly.

## Architectural Deviations

- Spec or doc reference (e.g., docs/design/architecture.md#L...): `docs/contracts/plugin-protocol.md:1366-1372`.
- Observed divergence: Retryable result flag has no effect.
- Reason (if known): RetryManager wired only for exceptions.
- Alignment plan or decision needed: Decide if retryable results should be honored or remove the flag from protocol.

## Acceptance Criteria

- Retryable TransformResult errors are retried according to policy, or docs are updated to explicitly forbid this behavior.

## Tests

- Suggested tests to run: `pytest tests/engine/test_retry.py -k retryable_result`
- New tests required: Yes (retryable TransformResult).

## Notes / Links

- Related issues/PRs: N/A
- Related design docs: `docs/contracts/plugin-protocol.md`
