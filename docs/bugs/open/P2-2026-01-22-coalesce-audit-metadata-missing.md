# Bug Report: Coalesce merge metadata is computed but never recorded

## Summary

- Coalesce builds a rich `coalesce_metadata` payload (policy, branches, timing) but never persists it to the audit trail.
- Node state output only records `merged_into` and does not capture the merged row data or merge details.

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

1. Run a pipeline with a coalesce step.
2. Inspect node state output for the coalesce node and any audit tables for merge metadata.

## Expected Behavior

- Audit trail should include coalesce event details: input token IDs, policy used, wait duration, branches arrived, and merge strategy, along with merged output data.

## Actual Behavior

- `coalesce_metadata` is computed but discarded; output data only includes `merged_into`.

## Evidence

- Coalesce metadata is computed but only returned, not persisted: `src/elspeth/engine/coalesce_executor.py:252`
- Node state output only records `merged_into`: `src/elspeth/engine/coalesce_executor.py:244`
- Audit contract requires merge timing and strategy details: `docs/contracts/plugin-protocol.md#L1137`

## Impact

- User-facing impact: explain/replay cannot show how/when branches merged or which policy fired.
- Data integrity / security impact: missing audit data violates “input/output captured at every transform”.
- Performance or cost impact: none.

## Root Cause Hypothesis

- Coalesce executor returns metadata to the caller, but the caller ignores it and no audit persistence exists.

## Proposed Fix

- Code changes (modules/files):
  - Persist `coalesce_metadata` and merged output data in node state output or a dedicated coalesce audit table.
  - Update processor/orchestrator to store metadata returned in `CoalesceOutcome`.
- Config or schema changes: none (unless a new audit table is needed).
- Tests to add/update:
  - Add a test asserting coalesce metadata is present in audit records.
- Risks or migration steps:
  - Ensure storage size is acceptable; consider storing hashes if payload is large.

## Architectural Deviations

- Spec or doc reference (e.g., docs/design/architecture.md#L...): `docs/contracts/plugin-protocol.md#L1137`, `docs/design/subsystems/00-overview.md#L349`
- Observed divergence: coalesce audit details are not recorded.
- Reason (if known): metadata is computed but never written.
- Alignment plan or decision needed: decide audit storage location for coalesce event details.

## Acceptance Criteria

- Coalesce audit records include policy, branches arrived, wait duration, and merge strategy.
- Merged output data (or its hash) is recorded as the coalesce node output.

## Tests

- Suggested tests to run:
  - `pytest tests/engine/test_coalesce_executor.py -k audit`
- New tests required: yes (coalesce audit metadata recording)

## Notes / Links

- Related issues/PRs: N/A
- Related design docs: `docs/contracts/plugin-protocol.md`, `docs/design/subsystems/00-overview.md`
