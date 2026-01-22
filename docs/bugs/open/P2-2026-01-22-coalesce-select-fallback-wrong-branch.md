# Bug Report: Select merge strategy falls back to wrong branch when select_branch is missing

## Summary

- For `merge: select`, if the selected branch has not arrived, the executor silently falls back to the first arrived branch instead of failing or waiting for the selected branch.
- This violates the contract that `select` takes output from a specific branch only.

## Severity

- Severity: major
- Priority: P2

## Reporter

- Name or handle: codex
- Date: 2026-01-22
- Related run/issue ID: N/A

## Environment

- Commit/branch: `ae2c0e6f088f467276582fa8016f91b4d3bb26c7` (fix/rc1-bug-burndown-session-2)
- OS: Linux (Ubuntu kernel 6.8.0-90-generic)
- Python version: 3.13.1
- Config profile / env vars: N/A
- Data set or fixture: N/A

## Agent Context (if relevant)

- Goal or task prompt: deep dive into coalesce_executor, identify bugs, create bug docs
- Model/version: GPT-5 (Codex CLI)
- Tooling and permissions (sandbox/approvals): workspace-write, network restricted, approvals on-request
- Determinism details (seed, run ID): N/A
- Notable tool calls or steps: code inspection of coalesce executor

## Steps To Reproduce

1. Configure coalesce with `merge: select` and `select_branch: preferred`.
2. Use a policy that allows partial arrival (e.g., `best_effort` timeout) and ensure `preferred` never arrives.
3. Observe the merged output.

## Expected Behavior

- If `select_branch` is missing, the merge should fail or wait (per policy), not substitute another branch.

## Actual Behavior

- The executor silently returns the first arrived branch output.

## Evidence

- Fallback to first arrival when select branch missing: `src/elspeth/engine/coalesce_executor.py:297`
- `select` is defined as “take output from specific branch only”: `docs/contracts/plugin-protocol.md#L1105`

## Impact

- User-facing impact: merged outputs can contain data from the wrong branch.
- Data integrity / security impact: audit trail cannot prove the intended branch was used.
- Performance or cost impact: none.

## Root Cause Hypothesis

- A convenience fallback was added to avoid missing-branch errors, but it violates the `select` contract.

## Proposed Fix

- Code changes (modules/files):
  - Remove the fallback and treat missing `select_branch` as a failure (or hold until arrival if policy permits).
  - Record failure or timeout appropriately when `select_branch` never arrives.
- Config or schema changes: none.
- Tests to add/update:
  - Add a test that `merge: select` fails when `select_branch` is missing.
- Risks or migration steps:
  - Existing pipelines relying on fallback (if any) will fail fast; document behavior change.

## Architectural Deviations

- Spec or doc reference (e.g., docs/design/architecture.md#L...): `docs/contracts/plugin-protocol.md#L1105`
- Observed divergence: select branch can be replaced by another branch.
- Reason (if known): fallback implemented for convenience.
- Alignment plan or decision needed: enforce strict select semantics.

## Acceptance Criteria

- Coalesce with `merge: select` never emits data from a non-selected branch.
- Missing selected branch results in a recorded failure or timeout handling.

## Tests

- Suggested tests to run:
  - `pytest tests/engine/test_coalesce_executor.py -k select`
- New tests required: yes (select branch missing)

## Notes / Links

- Related issues/PRs: N/A
- Related design docs: `docs/contracts/plugin-protocol.md`
