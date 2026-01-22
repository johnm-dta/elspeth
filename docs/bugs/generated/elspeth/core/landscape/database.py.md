# Bug Report: Schema validation misses newer required columns

## Summary

- `_validate_schema()` only checks `_REQUIRED_COLUMNS` (currently just `tokens.expand_group_id`), so stale SQLite databases missing newer required columns (e.g., `nodes.schema_mode`, `nodes.schema_fields_json`) pass validation and then crash later during inserts like `register_node`, defeating the intended early, clear compatibility error.

## Severity

- Severity: major
- Priority: P2

## Reporter

- Name or handle: codex
- Date: 2026-01-22
- Related run/issue ID: Unknown

## Environment

- Commit/branch: `81a0925d7d6de0d0e16fdd2d535f63d096a7d052` on `fix/rc1-bug-burndown-session-2`
- OS: Unknown
- Python version: Unknown
- Config profile / env vars: Unknown
- Data set or fixture: Unknown

## Agent Context (if relevant)

- Goal or task prompt: Static analysis agent doing deep bug audit for `src/elspeth/core/landscape/database.py`
- Model/version: GPT-5 (Codex CLI)
- Tooling and permissions (sandbox/approvals): read-only sandbox, approvals disabled
- Determinism details (seed, run ID): Unknown
- Notable tool calls or steps: `cat CLAUDE.md`, `rg`, `sed`, `nl -ba` on schema/recorder/database

## Steps To Reproduce

1. Create a SQLite DB with an older `nodes` table that lacks `schema_mode` and `schema_fields_json` (pre-WP-11.99 schema).
2. Initialize `LandscapeDB` with that DB; `_validate_schema()` passes because those columns are not in `_REQUIRED_COLUMNS`.
3. Call `LandscapeRecorder.register_node(...)` which inserts `schema_mode`/`schema_fields_json`; observe SQL error for missing columns.

## Expected Behavior

- Stale SQLite schemas missing required columns fail early with `SchemaCompatibilityError` before any inserts.

## Actual Behavior

- Validation passes, then runtime SQL errors occur later when code writes to missing columns.

## Evidence

- Logs or stack traces: Unknown (static analysis)
- Artifacts (paths, IDs, screenshots): Unknown
- Minimal repro input (attach or link): Unknown
- Code references:
  - `_REQUIRED_COLUMNS` only checks `tokens.expand_group_id` in `src/elspeth/core/landscape/database.py:24` and `_validate_schema()` uses it in `src/elspeth/core/landscape/database.py:84`
  - New schema columns `schema_mode` and `schema_fields_json` are defined in `src/elspeth/core/landscape/schema.py:63`
  - `register_node()` inserts `schema_mode` and `schema_fields_json` in `src/elspeth/core/landscape/recorder.py:516`
  - Design docs show these columns were added later (WP-11.99) in `docs/plans/completed/plugin-refactor/work-packages.md:575`

## Impact

- User-facing impact: Pipelines can fail mid-run with opaque SQL errors when a developer uses a stale local SQLite audit DB.
- Data integrity / security impact: No corruption, but audit trail is incomplete due to failed inserts.
- Performance or cost impact: Wasted processing time before failure; developer time lost diagnosing schema mismatch.

## Root Cause Hypothesis

- `_REQUIRED_COLUMNS` is not kept in sync with newly added required columns, so `_validate_schema()` misses stale schemas.

## Proposed Fix

- Code changes (modules/files):
  - Update `_REQUIRED_COLUMNS` in `src/elspeth/core/landscape/database.py` to include `nodes.schema_mode` and `nodes.schema_fields_json` (and any other columns added post-initial schema such as run export-tracking fields if applicable), or replace the list with a metadata-driven “missing column” check for existing tables.
- Config or schema changes: None.
- Tests to add/update:
  - Add a test similar to `test_old_schema_missing_column_fails_validation` that creates a `nodes` table without `schema_mode`/`schema_fields_json` and asserts `SchemaCompatibilityError`.
- Risks or migration steps:
  - This will force stale local SQLite DBs to be migrated or deleted (expected and consistent with current error messaging).

## Architectural Deviations

- Spec or doc reference (e.g., docs/design/architecture.md#L...): Unknown
- Observed divergence: Schema validation does not enforce all required audit columns.
- Reason (if known): Missing updates to `_REQUIRED_COLUMNS` when new columns were introduced.
- Alignment plan or decision needed: Expand validation coverage or derive required columns from `metadata`.

## Acceptance Criteria

- Opening a SQLite DB missing `nodes.schema_mode` or `nodes.schema_fields_json` raises `SchemaCompatibilityError` before any inserts.
- Error message lists the missing columns and remediation steps.

## Tests

- Suggested tests to run: `pytest tests/core/landscape/test_database.py -k schema`
- New tests required: Yes (missing nodes schema columns)

## Notes / Links

- Related issues/PRs: Unknown
- Related design docs: `docs/plans/completed/plugin-refactor/work-packages.md:575`
