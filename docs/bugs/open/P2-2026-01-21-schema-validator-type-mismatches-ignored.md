# Bug Report: schema_validator ignores type mismatches between producer and consumer schemas

## Summary

- `validate_pipeline_schemas` only checks for missing required field names.
- It does not detect incompatible field types (e.g., producer `value: str`, consumer `value: int`).
- This allows pipelines with incompatible types to pass validation and fail later at runtime.

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
- Data set or fixture: any pipeline with mismatched field types

## Agent Context (if relevant)

- Goal or task prompt: deep dive into `src/elspeth/engine/schema_validator.py`, identify bugs, create tickets
- Model/version: GPT-5 (Codex CLI)
- Tooling and permissions (sandbox/approvals): workspace-write, network restricted, approvals on-request
- Determinism details (seed, run ID): N/A
- Notable tool calls or steps: code inspection (no runtime execution)

## Steps To Reproduce

1. Define a producer schema with `value: str`.
2. Define a consumer schema with `value: int` (required).
3. Call `validate_pipeline_schemas(...)` with those schemas (or run a pipeline).

## Expected Behavior

- Schema validation rejects incompatible field types with a clear error message.

## Actual Behavior

- Validation passes because only field names are checked.

## Evidence

- `_get_missing_required_fields` checks only field names: `src/elspeth/engine/schema_validator.py:80-96`
- Type compatibility logic exists but is unused: `src/elspeth/contracts/data.py:131-205`

## Impact

- User-facing impact: pipelines can start with incompatible types and fail mid-run in transforms/sinks.
- Data integrity / security impact: violates Tier 2 rule (wrong types are upstream bugs) by letting incompatible schemas through.
- Performance or cost impact: reruns and debugging time.

## Root Cause Hypothesis

- Schema validator was implemented as a missing-field check and never upgraded to use the richer compatibility logic in `contracts.data.check_compatibility`.

## Proposed Fix

- Code changes (modules/files):
  - `src/elspeth/engine/schema_validator.py`: use `check_compatibility` (or similar) to detect type mismatches in addition to missing fields.
- Config or schema changes: none.
- Tests to add/update:
  - Add a test in `tests/engine/test_schema_validator.py` asserting a type mismatch is reported.
- Risks or migration steps:
  - Pipelines that previously passed validation may now be rejected; document as correctness fix.

## Architectural Deviations

- Spec or doc reference: `CLAUDE.md` Tier 2 rules (wrong types are upstream bugs)
- Observed divergence: type incompatibilities are not validated at construction time.
- Reason (if known): schema validator uses name-only checks.
- Alignment plan or decision needed: decide whether to enforce full type compatibility at build time.

## Acceptance Criteria

- Type mismatches between producer and consumer schemas are detected and reported.

## Tests

- Suggested tests to run: `pytest tests/engine/test_schema_validator.py`
- New tests required: yes

## Notes / Links

- Related issues/PRs: N/A
- Related design docs: `docs/contracts/plugin-protocol.md`
