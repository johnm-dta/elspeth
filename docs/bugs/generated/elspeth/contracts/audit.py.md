# Bug Report: BatchOutput contract missing batch_output_id

## Summary

- The `BatchOutput` contract omits the `batch_output_id` primary key defined in the `batch_outputs` table, so the contract cannot faithfully represent or round-trip table rows, undermining audit traceability when batch outputs are persisted or queried.

## Severity

- Severity: major
- Priority: P2

## Reporter

- Name or handle: Codex
- Date: 2026-01-22
- Related run/issue ID: Unknown

## Environment

- Commit/branch: 81a0925d7d6de0d0e16fdd2d535f63d096a7d052 / fix/rc1-bug-burndown-session-2
- OS: Linux nyx.foundryside.dev 6.8.0-90-generic #91-Ubuntu SMP PREEMPT_DYNAMIC Tue Nov 18 14:14:30 UTC 2025 x86_64 GNU/Linux
- Python version: Python 3.13.1
- Config profile / env vars: Unknown
- Data set or fixture: Unknown

## Agent Context (if relevant)

- Goal or task prompt: Static analysis bug audit for `src/elspeth/contracts/audit.py`
- Model/version: GPT-5 (Codex)
- Tooling and permissions (sandbox/approvals): Read-only filesystem sandbox; approvals disabled
- Determinism details (seed, run ID): Unknown
- Notable tool calls or steps: Inspected `src/elspeth/contracts/audit.py` and `src/elspeth/core/landscape/schema.py`

## Steps To Reproduce

1. Inspect the `batch_outputs` table definition and note the required `batch_output_id` primary key.
2. Inspect the `BatchOutput` contract and observe it lacks `batch_output_id`.
3. Attempt to map or persist a `batch_outputs` row using the contract; the primary key has no representation.

## Expected Behavior

- `BatchOutput` includes `batch_output_id` so audit rows can be fully represented and uniquely identified.

## Actual Behavior

- `BatchOutput` drops `batch_output_id`, preventing round-trip fidelity and unique identification of batch output rows.

## Evidence

- Logs or stack traces: Unknown
- Artifacts (paths, IDs, screenshots): `src/elspeth/contracts/audit.py:285`, `src/elspeth/core/landscape/schema.py:272`
- Minimal repro input (attach or link): Unknown

## Impact

- User-facing impact: Potential inability to reference or export unique batch output records once batch outputs are recorded.
- Data integrity / security impact: Loss of primary key in the contract breaks strict audit trail fidelity for `batch_outputs`.
- Performance or cost impact: Unknown

## Root Cause Hypothesis

- `BatchOutput` was defined without the `batch_output_id` field despite the schema requiring it as a primary key.

## Proposed Fix

- Code changes (modules/files): Add `batch_output_id: str` to `BatchOutput` in `src/elspeth/contracts/audit.py`.
- Config or schema changes: None.
- Tests to add/update: Update `tests/contracts/test_audit.py` to require `batch_output_id` for `BatchOutput` and add a contract/schema parity check if desired.
- Risks or migration steps: If any callers already construct `BatchOutput`, add a default or update call sites to provide the ID.

## Architectural Deviations

- Spec or doc reference (e.g., docs/design/architecture.md#L...): `src/elspeth/contracts/audit.py:1`
- Observed divergence: Contract claims to model Landscape tables but omits `batch_output_id` from `batch_outputs`.
- Reason (if known): Unknown
- Alignment plan or decision needed: Align `BatchOutput` contract with schema by including the primary key.

## Acceptance Criteria

- `BatchOutput` includes `batch_output_id` and any creation/mapping code can round-trip `batch_outputs` rows without losing the primary key.

## Tests

- Suggested tests to run: `pytest tests/contracts/test_audit.py::TestBatchOutput`
- New tests required: A contract/schema parity test for `batch_outputs` (optional).

## Notes / Links

- Related issues/PRs: Unknown
- Related design docs: Unknown
