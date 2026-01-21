# Bug Report: DatabaseSink rejects extra fields in free/dynamic schemas

## Summary

- DatabaseSink creates table columns from explicit schema fields (or first row for dynamic), so rows with extra fields permitted by free/dynamic schemas cause SQLAlchemy insert errors (unconsumed column names).

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

1. Configure DatabaseSink with schema `{mode: "free", fields: ["id: int"]}`.
2. Write rows that include an extra field, e.g. `{id: 1, extra: "x"}`.
3. Observe SQLAlchemy `ArgumentError` for unconsumed column names.

## Expected Behavior

- Free or dynamic schemas should accept extra fields or explicitly reject them up front with a clear configuration error.

## Actual Behavior

- Inserts fail at runtime because the table does not include extra columns.

## Evidence

- `SchemaConfig.allows_extra_fields` returns true for dynamic/free.
- `src/elspeth/plugins/sinks/database_sink.py` creates columns from schema fields or first row only and then calls `insert(self._table)` with full row dicts.

## Impact

- User-facing impact: Pipelines crash when valid rows contain extra fields.
- Data integrity / security impact: Output cannot be persisted despite schema allowing extras.
- Performance or cost impact: Runtime failures after partial processing.

## Root Cause Hypothesis

- DatabaseSink does not handle `allows_extra_fields` and uses a fixed table schema.

## Proposed Fix

- Code changes (modules/files):
  - If extras are allowed, either (a) extend the table schema dynamically, (b) strip extras before insert, or (c) fail fast for free/dynamic schemas with a clear error.
- Config or schema changes: None.
- Tests to add/update:
  - Add tests for free/dynamic schemas with extra fields and define expected behavior.
- Risks or migration steps: Dynamic schema evolution may require migrations or explicit opt-in.

## Architectural Deviations

- Spec or doc reference: `src/elspeth/contracts/schema.py` (`allows_extra_fields`).
- Observed divergence: Extras allowed by schema but rejected by sink.
- Reason (if known): Table schema built once, no evolution.
- Alignment plan or decision needed: Decide how DatabaseSink should handle extra fields.

## Acceptance Criteria

- Rows with extra fields under free/dynamic schemas either insert successfully or fail fast with a clear, documented configuration error.

## Tests

- Suggested tests to run: `pytest tests/plugins/sinks/test_database_sink.py -k free`
- New tests required: Free/dynamic extra-field handling.

## Notes / Links

- Related issues/PRs: N/A
- Related design docs: N/A
