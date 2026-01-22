# Bug Report: JSONSource accepts NaN/Infinity constants, leading to canonicalization crashes

## Summary

- Python's `json.load` / `json.loads` accept non-standard constants (`NaN`, `Infinity`, `-Infinity`) by default.
- `JSONSource` uses the default parser, so non-finite floats can enter pipeline rows (especially with dynamic schemas or `any` fields).
- Canonical hashing explicitly rejects non-finite floats, so these rows can crash later during hashing or audit recording.

## Severity

- Severity: major
- Priority: P2

## Reporter

- Name or handle: codex
- Date: 2026-01-21
- Related run/issue ID: N/A (static analysis)

## Environment

- Commit/branch: `main` @ `ae2c0e6f088f467276582fa8016f91b4d3bb26c7`
- OS: Linux 6.8.0-90-generic
- Python version: 3.13.1
- Config profile / env vars: N/A
- Data set or fixture: JSON/JSONL containing `NaN` or `Infinity` tokens

## Agent Context (if relevant)

- Goal or task prompt: deep dive into `src/elspeth/plugins/sources`, identify bugs, create tickets
- Model/version: GPT-5 (Codex CLI)
- Tooling and permissions (sandbox/approvals): workspace-write, network restricted, approvals on-request
- Determinism details (seed, run ID): N/A
- Notable tool calls or steps: code inspection (no runtime execution)

## Steps To Reproduce

1. Create a JSONL file containing a non-standard constant, e.g. `{"score": NaN}`.
2. Configure `JSONSource` with `schema: { fields: dynamic }` (or any field typed as `any`) and `on_validation_failure: quarantine`.
3. Run the pipeline.

## Expected Behavior

- Non-finite constants are rejected at the source boundary and quarantined (or discarded), per canonical JSON policy.

## Actual Behavior

- `json.loads` accepts the value and yields `float("nan")`.
- Dynamic schemas allow the row through; later hashing (`stable_hash` / `canonical_json`) raises `ValueError` for non-finite floats.

## Evidence

- JSON parsing uses default `json.load`/`json.loads` without `parse_constant`: `src/elspeth/plugins/sources/json_source.py:121-170`
- Canonicalization rejects NaN/Infinity: `src/elspeth/core/canonical.py:30-56`
- Hashing is invoked throughout execution (`stable_hash`): `src/elspeth/engine/executors.py:156-175`

## Impact

- User-facing impact: pipelines crash on inputs that should be quarantined.
- Data integrity / security impact: violates canonicalization policy and Tier 3 boundary handling.
- Performance or cost impact: reruns and manual sanitization required.

## Root Cause Hypothesis

- JSON parsing accepts non-standard constants and dynamic schemas do not enforce finiteness, so NaN/Infinity reaches hashing.

## Proposed Fix

- Code changes (modules/files):
  - Pass `parse_constant` to `json.load`/`json.loads` to reject `NaN`/`Infinity` at parse time (raise a controlled error).
  - Treat these parse failures as validation errors and quarantine/discard per `on_validation_failure`.
- Config or schema changes: none.
- Tests to add/update:
  - Add JSON and JSONL tests that include `NaN`/`Infinity` and assert quarantine instead of crash.
- Risks or migration steps:
  - Behavior change for users relying on non-standard JSON; document as invalid external data per audit policy.

## Architectural Deviations

- Spec or doc reference: `CLAUDE.md` (Tier 3 handling; canonical JSON strictness)
- Observed divergence: non-finite values can pass source parsing and crash later.
- Reason (if known): default JSON parser allows non-standard constants.
- Alignment plan or decision needed: enforce strict JSON parsing at the source boundary.

## Acceptance Criteria

- `NaN`/`Infinity` tokens are rejected at parse time and handled via quarantine/discard.
- No downstream hashing/canonicalization crashes from non-finite values.

## Tests

- Suggested tests to run: `pytest tests/plugins/sources/test_json_source.py`
- New tests required: yes

## Notes / Links

- Related issues/PRs: `docs/bugs/closed/P2-2026-01-19-non-finite-floats-pass-source-validation.md`
- Related design docs: `CLAUDE.md`
