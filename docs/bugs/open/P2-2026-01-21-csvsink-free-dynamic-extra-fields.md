# Bug Report: CSVSink rejects extra fields in free/dynamic schemas

## Summary

- For `schema.mode=free` or dynamic schemas, extra fields are allowed by the schema contract, but CSVSink uses a fixed header and `csv.DictWriter` defaults to `extrasaction="raise"`, causing valid rows to crash when they include extra fields.

## Severity

- Severity: major
- Priority: P2

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

1. Configure CSVSink with schema `{mode: "free", fields: ["id: int"]}` and `validate_input: true`.
2. Write a row `{"id": 1, "extra": "x"}`.
3. Observe `ValueError: dict contains fields not in fieldnames` even though the schema allows extras.

## Expected Behavior

- Free or dynamic schemas should either include extra fields in the CSV output or ignore them without crashing.

## Actual Behavior

- CSVSink raises when rows contain extra keys that are permitted by the schema.

## Evidence

- `SchemaConfig.allows_extra_fields` returns true for dynamic/free.
- `src/elspeth/plugins/sinks/csv_sink.py` initializes `csv.DictWriter` without `extrasaction="ignore"` and uses fixed fieldnames.

## Impact

- User-facing impact: Valid rows crash the sink when additional fields appear.
- Data integrity / security impact: Output may be incomplete or pipeline fails despite valid schema.
- Performance or cost impact: Unplanned failures mid-run.

## Root Cause Hypothesis

- CSVSink does not handle `allows_extra_fields` and always enforces a fixed header.

## Proposed Fix

- Code changes (modules/files):
  - If schema allows extra fields, set `extrasaction="ignore"` or dynamically expand headers (define policy).
- Config or schema changes: None.
- Tests to add/update:
  - Add tests for free/dynamic schemas with extra fields.
- Risks or migration steps: If choosing to ignore extras, document that behavior.

## Architectural Deviations

- Spec or doc reference: `src/elspeth/contracts/schema.py` (`allows_extra_fields`).
- Observed divergence: Extras allowed by schema but rejected by sink.
- Reason (if known): Fixed-header implementation.
- Alignment plan or decision needed: Decide how CSV should represent extra fields.

## Acceptance Criteria

- Rows with extra fields under free/dynamic schemas do not raise during write.

## Tests

- Suggested tests to run: `pytest tests/plugins/sinks/test_csv_sink.py -k free`
- New tests required: Free/dynamic extra-field handling.

## Notes / Links

- Related issues/PRs: N/A
- Related design docs: N/A
