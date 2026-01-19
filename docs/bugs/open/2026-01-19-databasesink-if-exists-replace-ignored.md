# Bug Report: DatabaseSink config `if_exists="replace"` is accepted but ignored (no drop/replace behavior)

## Summary

- `DatabaseSinkConfig` supports `if_exists: "append" | "replace"` and `DatabaseSink` stores the value in `self._if_exists`.
- The value is never used; table creation always uses `create_all(..., checkfirst=True)` and inserts always append to the existing table.
- This makes `if_exists="replace"` misleading and can cause duplicate/accumulated outputs when users expect replacement.

## Severity

- Severity: major
- Priority: P2

## Reporter

- Name or handle: codex
- Date: 2026-01-19
- Related run/issue ID: N/A (static analysis)

## Environment

- Commit/branch: `main` @ `8cfebea78be241825dd7487fed3773d89f2d7079`
- OS: Linux (Ubuntu kernel 6.8.0-90-generic)
- Python version: 3.13.1
- Config profile / env vars: N/A
- Data set or fixture: N/A

## Agent Context (if relevant)

- Goal or task prompt: deep dive into subsystem 6 (plugins), identify bugs, create tickets
- Model/version: GPT-5.2 (Codex CLI)
- Tooling and permissions (sandbox/approvals): workspace-write, network restricted, approvals on-request
- Determinism details (seed, run ID): N/A
- Notable tool calls or steps: code inspection (no runtime execution)

## Steps To Reproduce

1. Configure `DatabaseSink` with `if_exists: "replace"`.
2. Run the pipeline twice against the same database/table.
3. Observe that the table contents include rows from both runs (append behavior), not replacement.

## Expected Behavior

- If `if_exists="replace"`, the sink should either:
  - drop and recreate the table (or truncate) before writing, or
  - fail fast stating that `"replace"` is not supported.

## Actual Behavior

- `"replace"` has no effect; behavior is always append.

## Evidence

- Config defines `if_exists` and sink stores it: `src/elspeth/plugins/sinks/database_sink.py:24-69`
- No other references to `_if_exists` / `if_exists` exist in implementation: `rg -n \"_if_exists|if_exists\" src/elspeth/plugins/sinks/database_sink.py`
- Table creation always uses `create_all(..., checkfirst=True)` (no drop): `src/elspeth/plugins/sinks/database_sink.py:91-108`

## Impact

- User-facing impact: users can unintentionally accumulate duplicate rows across runs.
- Data integrity / security impact: output DB state may not match declared config intent, undermining reproducibility.
- Performance or cost impact: larger tables and slower downstream queries due to duplicated data.

## Root Cause Hypothesis

- `if_exists` support was planned but not implemented; the sink uses a minimal “create if missing” table setup.

## Proposed Fix

- Code changes (modules/files):
  - Implement `replace` semantics:
    - For SQLite/Postgres/etc: `DROP TABLE` then recreate, or `TRUNCATE` (if schema stable).
    - Ensure this behavior is recorded in audit metadata (destructive).
  - Or, if replace is out-of-scope for RC-1, remove the config option and fail validation when it is provided.
- Config or schema changes:
  - Clarify whether “replace” means drop/recreate or truncate.
- Tests to add/update:
  - Add a test that configures `if_exists="replace"` and asserts the table is empty before write (or that the sink raises a clear error if unsupported).
- Risks or migration steps:
  - Drop/recreate is destructive; require explicit confirmation or document loudly.

## Architectural Deviations

- Spec or doc reference: N/A
- Observed divergence: config implies behavior that isn’t implemented.
- Reason (if known): incomplete implementation.
- Alignment plan or decision needed: decide whether DatabaseSink is allowed to perform destructive operations and how that should be audited.

## Acceptance Criteria

- `if_exists="replace"` either functions as documented or is rejected with a clear configuration error.

## Tests

- Suggested tests to run: `pytest tests/plugins/sinks/test_database_sink.py`
- New tests required: yes

## Notes / Links

- Related ticket: `docs/bugs/open/2026-01-19-databasesink-schema-inferred-from-first-row.md`
