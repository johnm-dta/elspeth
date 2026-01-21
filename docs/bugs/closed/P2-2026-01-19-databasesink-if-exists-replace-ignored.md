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

## Resolution

**Status:** CLOSED (2026-01-21)
**Resolved by:** Claude Opus 4.5

### Changes Made

**Code fix (`src/elspeth/plugins/sinks/database_sink.py`):**

1. **Added `_table_replaced` tracking flag** (line 103):
   ```python
   self._table_replaced: bool = False  # Track if we've done the replace for this instance
   ```

2. **Modified `_ensure_table()` to check if_exists** (lines 122-126):
   ```python
   # Handle if_exists="replace": drop table on first write
   if self._if_exists == "replace" and not self._table_replaced:
       self._drop_table_if_exists()
       self._table_replaced = True
   ```

3. **Added `_drop_table_if_exists()` method** (lines 138-153):
   ```python
   def _drop_table_if_exists(self) -> None:
       """Drop the table if it exists (for replace mode)."""
       if self._engine is None:
           return

       from sqlalchemy import inspect, text

       inspector = inspect(self._engine)
       if inspector.has_table(self._table_name):
           with self._engine.begin() as conn:
               conn.execute(text(f'DROP TABLE "{self._table_name}"'))
   ```

**Tests added (`tests/plugins/sinks/test_database_sink.py`):**
- `TestDatabaseSinkIfExistsReplace` class with 4 regression tests:
  - `test_if_exists_replace_drops_existing_table` - verifies replace drops old data
  - `test_if_exists_replace_subsequent_writes_append` - verifies within-instance writes append
  - `test_if_exists_append_accumulates` - verifies default append behavior
  - `test_if_exists_replace_works_when_table_does_not_exist` - verifies no error when table missing

### Verification

```bash
.venv/bin/python -m pytest tests/plugins/sinks/test_database_sink.py -v
# 19 passed (15 existing + 4 new)
```

### Notes

The implementation follows pandas `to_sql` semantics:
- `if_exists="replace"` drops the table on first write of each sink instance
- Subsequent writes within the same instance append (like pandas behavior)
- `if_exists="append"` (default) accumulates across sink instances

Uses `inspect().has_table()` + raw SQL `DROP TABLE` instead of `MetaData.reflect()` to avoid errors when table doesn't exist.
