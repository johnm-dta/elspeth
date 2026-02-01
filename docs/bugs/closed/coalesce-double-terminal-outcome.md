# Bug Report: Coalesce select-merge failure double-records terminal outcome

## Summary

- When a coalesce uses `merge="select"` and the selected branch has not arrived, `CoalesceExecutor` already records `FAILED` outcomes for all arrived tokens, but `RowProcessor._maybe_coalesce_token` records `FAILED` again for the current token, violating the unique terminal outcome constraint and crashing the run.

## Severity

- Severity: major
- Priority: P1

## Reporter

- Name or handle: Codex
- Date: 2026-01-30
- Related run/issue ID: N/A

## Environment

- Commit/branch: aa9cf1467c66d6e52fc2b87c1eea55fa7e42f304 (fix/P2-aggregation-metadata-hardcoded)
- OS: unknown
- Python version: unknown
- Config profile / env vars: N/A
- Data set or fixture: coalesce with merge=select and policy allowing early merge (e.g., first/quorum)

## Agent Context (if relevant)

- Goal or task prompt: Static analysis deep bug audit of /home/john/elspeth-rapid/src/elspeth/engine/processor.py
- Model/version: Codex (GPT-5)
- Tooling and permissions (sandbox/approvals): read-only sandbox
- Determinism details (seed, run ID): N/A
- Notable tool calls or steps: code review only

## Steps To Reproduce

1. Configure a gate that forks into branches `A` and `B`, and a coalesce with `merge="select"`, `select_branch="A"`, and policy `first` (or `quorum`) so merge can trigger without all branches.
2. Run a row so that branch `B` arrives at the coalesce before `A`.
3. Observe the coalesce failure path (`select_branch_not_arrived`) and subsequent handling.

## Expected Behavior

- The coalesce failure should record a single terminal `FAILED` outcome for each arrived token and return a failure result without crashing.

## Actual Behavior

- The coalesce failure records `FAILED` outcomes in `CoalesceExecutor`, then `RowProcessor._maybe_coalesce_token` records `FAILED` again for the current token, violating the unique terminal-outcome constraint and causing a crash.

## Evidence

- `src/elspeth/engine/processor.py:1506-1515` records `FAILED` on any `coalesce_outcome.failure_reason`, unconditionally.
- `src/elspeth/engine/coalesce_executor.py:315-340` already records `FAILED` outcomes for all arrived tokens on `select_branch_not_arrived`.
- `src/elspeth/core/landscape/schema.py:156-163` enforces exactly one terminal outcome per token via a unique partial index.

## Impact

- User-facing impact: pipeline crashes during coalesce failure scenarios instead of returning a row-scoped failure.
- Data integrity / security impact: attempted double terminal-outcome insert violates audit invariants; audit trail can become incomplete if the run aborts.
- Performance or cost impact: failed runs and retries increase processing time and operational cost.

## Root Cause Hypothesis

- `RowProcessor._maybe_coalesce_token` does not distinguish coalesce failure modes where the executor already recorded terminal outcomes, so it attempts to record a duplicate `FAILED` outcome for the same token.

## Proposed Fix

- Code changes (modules/files):
  - `src/elspeth/engine/processor.py`: In `_maybe_coalesce_token`, skip `record_token_outcome` when `failure_reason == "select_branch_not_arrived"` (or add an explicit flag in `CoalesceOutcome` to indicate outcomes already recorded).
- Config or schema changes: None.
- Tests to add/update:
  - Add a coalesce integration test for `merge="select"` with policy `first` (or `quorum`) where the selected branch does not arrive first; assert no duplicate terminal outcome and no crash.
- Risks or migration steps:
  - Low risk; scoped to coalesce failure handling.

## Architectural Deviations

- Spec or doc reference (e.g., docs/design/architecture.md#L...): `CLAUDE.md` (“Every row reaches exactly one terminal state”) and `src/elspeth/core/landscape/schema.py:156-163` (unique terminal outcome per token).
- Observed divergence: duplicate terminal outcome recording on select-merge failure.
- Reason (if known): processor failure handling is unaware that executor already recorded terminal outcomes for this failure mode.
- Alignment plan or decision needed: ensure processor only records terminal outcomes when they have not already been recorded by the coalesce executor.

## Acceptance Criteria

- Coalesce `select_branch_not_arrived` failure completes without unique-constraint violations.
- Exactly one terminal outcome is recorded per token in all coalesce failure modes.
- New test passes demonstrating no crash on select-merge failure.

## Tests

- Suggested tests to run: `.venv/bin/python -m pytest tests/engine/test_coalesce_integration.py -k select`
- New tests required: yes, coalesce select-merge failure path

## Notes / Links

- Related issues/PRs: N/A
- Related design docs: `CLAUDE.md` (Terminal Row States), `docs/contracts/plugin-protocol.md` (coalesce semantics)
