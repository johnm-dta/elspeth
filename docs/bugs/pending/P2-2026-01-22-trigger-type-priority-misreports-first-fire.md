# Bug Report: Trigger type misreported when multiple triggers are true

## Summary

- `TriggerEvaluator.should_trigger()` checks count, then timeout, then condition. When multiple triggers are true at evaluation time, it always reports the first in that fixed order, even if another trigger would have fired earlier. This violates the "first one to fire wins" contract and mislabels `trigger_type` in the audit trail.

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
- Notable tool calls or steps: inspected trigger ordering and audit usage

## Steps To Reproduce

1. Configure an aggregation trigger with `count: 100` and `timeout_seconds: 1`.
2. Accept 99 rows quickly (<1s), then wait >1s to exceed the timeout.
3. Accept the 100th row; `should_trigger()` returns `True` with `which_triggered() == "count"`.

## Expected Behavior

- `TriggerType.TIMEOUT` should be recorded because the timeout elapsed before the count threshold was reached.

## Actual Behavior

- `TriggerType.COUNT` is recorded due to fixed ordering in `should_trigger()`.

## Evidence

- Fixed trigger order: `src/elspeth/engine/triggers.py:95-116`
- "First one to fire wins" contract: `docs/contracts/plugin-protocol.md:1213`

## Impact

- User-facing impact: misleading trigger type in batch audit metadata.
- Data integrity / security impact: audit trail attributes batches to the wrong trigger.
- Performance or cost impact: none directly, but mislabels can obscure tuning.

## Root Cause Hypothesis

- Trigger evaluator does not track when each condition first becomes true; it short-circuits on a fixed priority order.

## Proposed Fix

- Code changes (modules/files):
  - Track the timestamp when each trigger first becomes true.
  - When multiple triggers are satisfied, select the one with the earliest fire time.
- Config or schema changes: none.
- Tests to add/update:
  - Add a test where timeout elapses before count threshold; assert trigger_type is TIMEOUT.
- Risks or migration steps:
  - Ensure the chosen trigger type remains stable across re-evaluations.

## Architectural Deviations

- Spec or doc reference: `docs/contracts/plugin-protocol.md:1213`
- Observed divergence: fixed priority order contradicts "first to fire wins."
- Reason (if known): order-based short-circuiting for simplicity.
- Alignment plan or decision needed: define tie-breaking rules for simultaneous triggers.

## Acceptance Criteria

- When multiple triggers are true, the recorded trigger type reflects the earliest trigger to fire.

## Tests

- Suggested tests to run: `pytest tests/engine/test_triggers.py -k combined`
- New tests required: yes

## Notes / Links

- Related issues/PRs: N/A
- Related design docs: `docs/contracts/plugin-protocol.md`
