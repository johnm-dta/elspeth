# Bug Report: schema_validator ignores strict-schema extra-field constraints

## Summary

- Strict schemas (extra="forbid") reject unknown fields, but the schema validator only checks that required fields exist.
- A producer schema that includes additional fields can pass validation even when a strict consumer forbids extras.
- This creates false positives where the pipeline validates but strict sinks/transforms reject rows at runtime.

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
- Data set or fixture: any strict consumer schema with a producer that declares extra fields

## Agent Context (if relevant)

- Goal or task prompt: deep dive into `src/elspeth/engine/schema_validator.py`, identify bugs, create tickets
- Model/version: GPT-5 (Codex CLI)
- Tooling and permissions (sandbox/approvals): workspace-write, network restricted, approvals on-request
- Determinism details (seed, run ID): N/A
- Notable tool calls or steps: code inspection (no runtime execution)

## Steps To Reproduce

1. Create a strict consumer schema (extra="forbid") with fields `id, name`.
2. Create a producer schema that includes `id, name, extra_field`.
3. Call `validate_pipeline_schemas(...)` with these schemas.

## Expected Behavior

- Validation fails because strict consumers do not accept extra fields declared by the producer.

## Actual Behavior

- Validation passes because only missing required fields are checked.

## Evidence

- Strict schemas set `extra="forbid"` in schema factory: `src/elspeth/plugins/schema_factory.py:74-116`
- Schema validator only checks required field names: `src/elspeth/engine/schema_validator.py:80-96`

## Impact

- User-facing impact: pipelines pass validation but strict sinks/transforms may reject rows at runtime.
- Data integrity / security impact: schema compatibility is overstated; audit metadata implies compatibility that does not hold for strict consumers.
- Performance or cost impact: reruns and manual debugging.

## Root Cause Hypothesis

- Validator does not consider `extra="forbid"` semantics when comparing producer and consumer schemas.

## Proposed Fix

- Code changes (modules/files):
  - Extend schema validation to check extra-field compatibility when consumer is strict.
  - If consumer forbids extras and producer declares additional fields, report incompatibility.
- Config or schema changes: none.
- Tests to add/update:
  - Add a test that strict consumer rejects producer schemas with extra fields.
- Risks or migration steps:
  - Existing pipelines with strict sinks may now fail validation; document as correctness fix.

## Architectural Deviations

- Spec or doc reference: `docs/contracts/plugin-protocol.md` (engine validates schema compatibility)
- Observed divergence: strict consumer constraints are ignored in compatibility checks.
- Reason (if known): validator only checks for missing required fields.
- Alignment plan or decision needed: define compatibility rules for strict schemas.

## Acceptance Criteria

- Strict consumers (extra="forbid") reject producer schemas that declare additional fields.

## Tests

- Suggested tests to run: `pytest tests/engine/test_schema_validator.py`
- New tests required: yes

## Notes / Links

- Related issues/PRs: N/A
- Related design docs: `src/elspeth/plugins/schema_factory.py`
