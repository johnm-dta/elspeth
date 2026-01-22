# Bug Report: Recovery skips rows for sinks written later due to row_index checkpoint boundary

## Summary

- RecoveryManager.get_unprocessed_rows uses the row_index of the latest checkpointed token as a single boundary; because checkpoints are created after sink writes in sink order, the latest checkpoint can correspond to an earlier row than some rows written to other sinks, causing resume to skip rows routed to a later/failed sink and leaving outputs missing.

## Severity

- Severity: critical
- Priority: P1

## Reporter

- Name or handle: Codex
- Date: 2026-01-22
- Related run/issue ID: Unknown

## Environment

- Commit/branch: Unknown
- OS: Unknown
- Python version: Unknown
- Config profile / env vars: Unknown
- Data set or fixture: Unknown

## Agent Context (if relevant)

- Goal or task prompt: Static analysis agent doing a deep bug audit of `src/elspeth/core/checkpoint/recovery.py`.
- Model/version: GPT-5 (Codex CLI)
- Tooling and permissions (sandbox/approvals): Read-only sandbox, approvals never.
- Determinism details (seed, run ID): Unknown
- Notable tool calls or steps: Read `src/elspeth/core/checkpoint/recovery.py`, `src/elspeth/core/checkpoint/manager.py`, `src/elspeth/engine/orchestrator.py`, `src/elspeth/engine/executors.py`, `src/elspeth/contracts/enums.py`, `CLAUDE.md`.

## Steps To Reproduce

1. Configure a pipeline with two sinks (`sink_a` default, `sink_b` via gate) and checkpoint frequency `every_row`.
2. Run with rows routed to both sinks; force `sink_b.write()` to raise after `sink_a` succeeds (simulate sink failure).
3. Call `RecoveryManager.get_unprocessed_rows(run_id)` and resume; observe rows routed to `sink_b` are not returned/resumed.

## Expected Behavior

- Recovery should include rows whose tokens never reached a completed sink node_state (including rows routed to `sink_b`), and resume should write the missing sink outputs.

## Actual Behavior

- `get_unprocessed_rows` uses the latest checkpoint’s token row_index and returns only rows with row_index greater than that, skipping rows routed to the failed/later sink and leaving their outputs missing.

## Evidence

- `src/elspeth/core/checkpoint/recovery.py:223`
- `src/elspeth/core/checkpoint/recovery.py:250`
- `src/elspeth/core/checkpoint/manager.py:93`
- `src/elspeth/engine/orchestrator.py:132`
- `src/elspeth/engine/orchestrator.py:885`
- `src/elspeth/engine/executors.py:1337`

## Impact

- User-facing impact: Resume can finish without emitting outputs for some sinks, even though the run reports completion.
- Data integrity / security impact: Audit trail implies sink outputs were produced, but artifacts are missing for routed rows; violates auditability guarantees.
- Performance or cost impact: Operators may rerun or manually backfill, risking duplicate writes and extra compute.

## Root Cause Hypothesis

- Recovery assumes a single monotonic row_index boundary derived from the latest checkpoint, but checkpoints are ordered by sequence_number (token write order) which is not aligned with row_index across multiple sinks; this causes rows routed to later/failed sinks to be skipped.

## Proposed Fix

- Code changes (modules/files): In `src/elspeth/core/checkpoint/recovery.py`, compute unprocessed rows by identifying tokens lacking a completed sink node_state (join `tokens_table`, `node_states_table`, and `nodes_table` where node_type="sink") and map those tokens to row_ids, instead of using a single row_index boundary from the latest checkpoint; optionally keep the current boundary as an optimization only when there is a single sink.
- Config or schema changes: None.
- Tests to add/update: Add a multi-sink recovery test where one sink fails after another succeeds; verify rows routed to the failed sink are returned by `get_unprocessed_rows`.
- Risks or migration steps: May reprocess some rows and create duplicate sink outputs if sinks are not idempotent; document expected behavior or add idempotency keys.

## Architectural Deviations

- Spec or doc reference (e.g., docs/design/architecture.md#L...): `CLAUDE.md:28`
- Observed divergence: Resume can mark runs completed while some sink outputs are missing for routed rows.
- Reason (if known): Recovery uses a row_index boundary based on the last checkpoint rather than actual sink completion per token.
- Alignment plan or decision needed: Decide whether recovery should be token/sink-state-based (accurate) or enforce per-row sink write ordering/checkpointing.

## Acceptance Criteria

- A multi-sink recovery test (with sink failure) returns rows routed to the failed sink and resume emits their outputs; no missing sink artifacts after resume.

## Tests

- Suggested tests to run: `tests/core/checkpoint/test_recovery.py`, `tests/integration/test_checkpoint_recovery.py`
- New tests required: Yes, multi-sink recovery scenario with a forced sink failure.

## Notes / Links

- Related issues/PRs: Unknown
- Related design docs: `CLAUDE.md:28`
---
# Bug Report: can_resume accepts invalid run status instead of failing fast

## Summary

- RecoveryManager.can_resume treats any run status other than RUNNING/COMPLETED as resumable, allowing resume on invalid run statuses and violating the audit DB “invalid enum value = crash” rule.

## Severity

- Severity: major
- Priority: P2

## Reporter

- Name or handle: Codex
- Date: 2026-01-22
- Related run/issue ID: Unknown

## Environment

- Commit/branch: Unknown
- OS: Unknown
- Python version: Unknown
- Config profile / env vars: Unknown
- Data set or fixture: Unknown

## Agent Context (if relevant)

- Goal or task prompt: Static analysis agent doing a deep bug audit of `src/elspeth/core/checkpoint/recovery.py`.
- Model/version: GPT-5 (Codex CLI)
- Tooling and permissions (sandbox/approvals): Read-only sandbox, approvals never.
- Determinism details (seed, run ID): Unknown
- Notable tool calls or steps: Read `src/elspeth/core/checkpoint/recovery.py`, `src/elspeth/contracts/enums.py`, `CLAUDE.md`.

## Steps To Reproduce

1. Insert or update a run with `runs.status = "paused"` (or any non-enum value) and create at least one checkpoint.
2. Call `RecoveryManager.can_resume(run_id)`.
3. Observe it returns `can_resume=True` (or proceeds without error) instead of rejecting the invalid status.

## Expected Behavior

- Invalid run statuses should raise or be rejected; only `RunStatus.FAILED` should be resumable.

## Actual Behavior

- Any status other than `running` or `completed` is treated as resumable if a checkpoint exists.

## Evidence

- `src/elspeth/core/checkpoint/recovery.py:103`
- `src/elspeth/core/checkpoint/recovery.py:109`
- `src/elspeth/contracts/enums.py:11`
- `CLAUDE.md:40`

## Impact

- User-facing impact: Resume may proceed on corrupted or invalid runs with unpredictable outcomes.
- Data integrity / security impact: Violates full-trust audit DB rule; corrupted status is not surfaced immediately.
- Performance or cost impact: Potential wasted compute on invalid runs.

## Root Cause Hypothesis

- can_resume only checks for RUNNING/COMPLETED and does not validate that status is a valid RunStatus or explicitly FAILED.

## Proposed Fix

- Code changes (modules/files): In `src/elspeth/core/checkpoint/recovery.py`, validate `run.status` against `RunStatus`; if it is not `RunStatus.FAILED`, return `can_resume=False` with reason or raise a RuntimeError on invalid enum values per audit policy.
- Config or schema changes: None.
- Tests to add/update: Add a test that inserts an invalid status and asserts can_resume raises or returns a non-resumable result.
- Risks or migration steps: None.

## Architectural Deviations

- Spec or doc reference (e.g., docs/design/architecture.md#L...): `CLAUDE.md:40`
- Observed divergence: Invalid enum values do not cause a crash or explicit failure in recovery checks.
- Reason (if known): Missing explicit enum validation in can_resume.
- Alignment plan or decision needed: Enforce strict enum validation for runs.status in recovery.

## Acceptance Criteria

- can_resume rejects or raises on any run.status not in RunStatus, and only FAILED is resumable.

## Tests

- Suggested tests to run: `tests/core/checkpoint/test_recovery.py`
- New tests required: Yes, invalid-status regression test.

## Notes / Links

- Related issues/PRs: Unknown
- Related design docs: `CLAUDE.md:40`
