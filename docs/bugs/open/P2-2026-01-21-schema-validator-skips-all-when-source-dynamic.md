# Bug Report: schema_validator skips all validation when source schema is None

## Summary

- `validate_pipeline_schemas` returns early if `source_output` is None.
- This bypasses validation for transform-to-transform and transform-to-sink compatibility even when those schemas are explicitly declared.
- As a result, common pipelines with dynamic sources get no schema validation at all.

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
- Data set or fixture: any pipeline with dynamic source and explicit downstream schemas

## Agent Context (if relevant)

- Goal or task prompt: deep dive into `src/elspeth/engine/schema_validator.py`, identify bugs, create tickets
- Model/version: GPT-5 (Codex CLI)
- Tooling and permissions (sandbox/approvals): workspace-write, network restricted, approvals on-request
- Determinism details (seed, run ID): N/A
- Notable tool calls or steps: code inspection (no runtime execution)

## Steps To Reproduce

1. Set `source_output=None` (dynamic source).
2. Provide incompatible transform and sink schemas (e.g., transform outputs `result: str`, sink requires `result: str` and `extra: int`).
3. Call `validate_pipeline_schemas(...)`.

## Expected Behavior

- Validation should still check compatibility between downstream stages that have explicit schemas.

## Actual Behavior

- Validation returns no errors because it exits early.

## Evidence

- Early return on dynamic source: `src/elspeth/engine/schema_validator.py:40-44`

## Impact

- User-facing impact: pipelines with dynamic sources receive no schema validation, even when downstream schemas are strict.
- Data integrity / security impact: missing early detection of incompatible transforms/sinks.
- Performance or cost impact: errors surface at runtime instead of build time.

## Root Cause Hypothesis

- The validator treats a dynamic source as reason to skip all checks, rather than only the source-to-first-transform edge.

## Proposed Fix

- Code changes (modules/files):
  - Only skip the source->first-transform validation when source schema is None.
  - Still validate transform chain and sinks when their schemas are explicit.
- Config or schema changes: none.
- Tests to add/update:
  - Add a test verifying downstream validation still runs with dynamic sources.
- Risks or migration steps:
  - Pipelines using dynamic sources may now fail validation if downstream schemas are incompatible.

## Architectural Deviations

- Spec or doc reference: `docs/contracts/plugin-protocol.md` (engine validates schema compatibility between connected nodes)
- Observed divergence: dynamic source disables all validation.
- Reason (if known): early return in validator.
- Alignment plan or decision needed: define validation scope when source is dynamic.

## Acceptance Criteria

- Downstream schema compatibility is validated even when the source schema is dynamic.

## Tests

- Suggested tests to run: `pytest tests/engine/test_schema_validator.py`
- New tests required: yes

## Notes / Links

- Related issues/PRs: N/A
- Related design docs: `docs/contracts/plugin-protocol.md`
