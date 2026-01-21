# Bug Report: CSVSink append mode ignores explicit schema when headers differ

## Summary

- In append mode, CSVSink reads existing CSV headers and uses them without validating against the configured explicit schema, allowing schema drift or late failures when new fields appear.

## Severity

- Severity: major
- Priority: P1

## Reporter

- Name or handle: Codex
- Date: 2026-01-21
- Related run/issue ID: N/A

## Environment

- Commit/branch: ae2c0e6f088f467276582fa8016f91b4d3bb26c7 (fix/rc1-bug-burndown-session-2)
- OS: Linux nyx.foundryside.dev 6.8.0-90-generic
- Python version: 3.13.1
- Config profile / env vars: N/A
- Data set or fixture: N/A

## Agent Context (if relevant)

- Goal or task prompt: Deep dive into src/elspeth/plugins/sinks for bugs.
- Model/version: Codex (GPT-5)
- Tooling and permissions (sandbox/approvals): Codex CLI, workspace-write sandbox
- Determinism details (seed, run ID): N/A
- Notable tool calls or steps: Manual code inspection only

## Steps To Reproduce

1. Create `output.csv` with header `id` (no `score` column).
2. Configure CSVSink with schema `{mode: "free", fields: ["id: int", "score: float?"]}` and `mode: "append"`.
3. Write a row that includes `score`.
4. Observe a runtime `ValueError` (extra field) or silent omission if data is coerced elsewhere.

## Expected Behavior

- Append mode should validate that existing headers match the configured explicit schema and fail early with a clear error if they do not.

## Actual Behavior

- Existing headers are accepted as authoritative, ignoring the explicit schema.

## Evidence

- `src/elspeth/plugins/sinks/csv_sink.py` reads `existing_fieldnames` and uses them directly in append mode without comparing to `schema_config.fields`.

## Impact

- User-facing impact: Append runs can fail late or silently drop schema-defined fields.
- Data integrity / security impact: Output can drift from the declared schema, undermining auditability.
- Performance or cost impact: Wasted run time before failure.

## Root Cause Hypothesis

- Append path prioritizes file headers over explicit schema, with no validation step.

## Proposed Fix

- Code changes (modules/files):
  - If schema is explicit, compare `existing_fieldnames` to schema field names and raise if mismatch.
  - Optionally allow a strict flag to force header rewrite (if allowed by policy).
- Config or schema changes: None.
- Tests to add/update:
  - Add append-mode test that asserts schema/header mismatch raises a clear error.
- Risks or migration steps: Existing append workflows with mismatched headers will start failing fast (desired).

## Architectural Deviations

- Spec or doc reference: `docs/contracts/plugin-protocol.md` (schema is a contract for sinks).
- Observed divergence: Explicit schema is ignored when appending to existing file.
- Reason (if known): Append mode implemented without schema validation.
- Alignment plan or decision needed: Enforce schema compliance on append.

## Acceptance Criteria

- Append mode fails fast with a clear error when file headers do not match explicit schema.

## Tests

- Suggested tests to run: `pytest tests/plugins/sinks/test_csv_sink_append.py -v`
- New tests required: Add schema mismatch coverage.

## Notes / Links

- Related issues/PRs: N/A
- Related design docs: N/A
