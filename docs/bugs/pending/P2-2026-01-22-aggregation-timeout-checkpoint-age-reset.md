# Bug Report: Timeout age resets after checkpoint restore

## Summary

- Crash recovery restores trigger counts by replaying `record_accept()`, which resets `first_accept_time` to the recovery time. Any elapsed batch age before the crash is lost, delaying timeout triggers beyond their configured window.

## Severity

- Severity: major
- Priority: P2

## Reporter

- Name or handle: codex
- Date: 2026-01-22
- Related run/issue ID: N/A

## Environment

- Commit/branch: `fix/rc1-bug-burndown-session-2` @ `ae2c0e6f088f467276582fa8016f91b4d3bb26c7`
- OS: Linux 6.8.0-90-generic
- Python version: 3.13.1
- Config profile / env vars: N/A
- Data set or fixture: N/A

## Agent Context (if relevant)

- Goal or task prompt: deep dive into `src/elspeth/engine/triggers.py` and file bugs
- Model/version: GPT-5 (Codex CLI)
- Tooling and permissions (sandbox/approvals): workspace-write, network restricted, approvals on-request
- Determinism details (seed, run ID): N/A
- Notable tool calls or steps: inspected checkpoint restore path for trigger state

## Steps To Reproduce

1. Configure an aggregation with `timeout_seconds: 10`.
2. Accept a row and wait ~9 seconds.
3. Simulate a crash, restore from checkpoint, then wait another ~2 seconds.
4. Observe that the batch does not flush even though total age exceeds 10 seconds.

## Expected Behavior

- Timeout should account for elapsed time before the crash and trigger shortly after recovery.

## Actual Behavior

- Timeout age is reset during recovery, effectively extending the timeout window.

## Evidence

- Recovery replays `record_accept()` without restoring timestamps: `src/elspeth/engine/executors.py:1072-1097`
- `record_accept()` initializes `first_accept_time` to `time.monotonic()`: `src/elspeth/engine/triggers.py:74-82`

## Impact

- User-facing impact: delayed batch flushes after recovery.
- Data integrity / security impact: trigger timing is inconsistent across crashes.
- Performance or cost impact: prolonged buffering increases memory usage.

## Root Cause Hypothesis

- Trigger evaluator lacks a persisted `first_accept_time` (or elapsed age) in checkpoint state.

## Proposed Fix

- Code changes (modules/files):
  - Include `first_accept_time` (or elapsed seconds) in checkpoint state.
  - Add a `restore` method on `TriggerEvaluator` to set count and age explicitly.
- Config or schema changes: extend checkpoint payload with trigger timing metadata.
- Tests to add/update:
  - Add checkpoint/restore test for timeout triggers preserving elapsed age.
- Risks or migration steps:
  - Define how to handle clock differences across process restarts.

## Architectural Deviations

- Spec or doc reference: `docs/contracts/plugin-protocol.md:1208-1210`
- Observed divergence: timeout semantics are not preserved through recovery.
- Reason (if known): checkpoint only stores buffered rows/token IDs.
- Alignment plan or decision needed: define timeout behavior across crashes.

## Acceptance Criteria

- Timeout-triggered flushes occur based on total elapsed batch age, including pre-crash time.

## Tests

- Suggested tests to run: `pytest tests/engine/test_executors.py -k checkpoint`
- New tests required: yes

## Notes / Links

- Related issues/PRs: N/A
- Related design docs: `docs/contracts/plugin-protocol.md`
