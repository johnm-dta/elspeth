# Bug Report: Resume path never calls transform.close()

## Summary

- _process_resumed_rows() calls on_complete for transforms but never calls close(), so transform resources (threads, clients, executors) are leaked during resume runs.

## Severity

- Severity: minor
- Priority: P3

## Reporter

- Name or handle: Codex
- Date: 2026-01-21
- Related run/issue ID: N/A

## Environment

- Commit/branch: ae2c0e6 (fix/rc1-bug-burndown-session-2)
- OS: unknown
- Python version: unknown
- Config profile / env vars: N/A
- Data set or fixture: resume runs with transforms that allocate resources

## Agent Context (if relevant)

- Goal or task prompt: deep dive into src/elspeth/engine/orchestrator.py
- Model/version: Codex (GPT-5)
- Tooling and permissions (sandbox/approvals): workspace-write sandbox, no escalations
- Determinism details (seed, run ID): N/A
- Notable tool calls or steps: code review only

## Steps To Reproduce

1. Configure a transform that allocates resources (e.g., pooled executor).
2. Run a pipeline, force a failure, and resume.
3. Observe that transform.close() is not called during resume cleanup.

## Expected Behavior

- Resume path should call transform.close() just like normal runs.

## Actual Behavior

- Only on_complete is called; close() is skipped in resume cleanup.

## Evidence

- Resume cleanup closes sinks only; no transform.close in `src/elspeth/engine/orchestrator.py:1424-1435`.
- Normal run cleanup calls transform.close via _cleanup_transforms().

## Impact

- User-facing impact: possible resource leaks or lingering threads after resume.
- Data integrity / security impact: none direct.
- Performance or cost impact: increased memory/CPU usage on repeated resumes.

## Root Cause Hypothesis

- Resume cleanup path omitted transform.close() call.

## Proposed Fix

- Code changes (modules/files):
  - Add transform.close() calls in _process_resumed_rows() finally block (mirror _cleanup_transforms()).
- Config or schema changes: N/A
- Tests to add/update:
  - Resume test that asserts transform.close() is invoked.
- Risks or migration steps:
  - Ensure close() is idempotent (contract already requires this).

## Architectural Deviations

- Spec or doc reference (e.g., docs/design/architecture.md#L...): TransformProtocol requires close() idempotency and lifecycle cleanup.
- Observed divergence: resume path omits close().
- Reason (if known): missing cleanup in resume path.
- Alignment plan or decision needed: standardize cleanup across run and resume.

## Acceptance Criteria

- Resumed runs invoke transform.close() for all transforms.

## Tests

- Suggested tests to run: `pytest tests/engine/test_orchestrator.py -k resume -v`
- New tests required: yes, resume cleanup test.

## Notes / Links

- Related issues/PRs: N/A
- Related design docs: CLAUDE.md lifecycle hooks
