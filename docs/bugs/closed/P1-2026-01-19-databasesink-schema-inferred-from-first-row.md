# Bug Report: DatabaseSink infers table columns from the first row, ignoring configured schema; later rows with additional (valid) fields can fail

## Summary

- `DatabaseSink` creates its table schema by inferring columns from the first row's keys.
- If later rows include additional keys (which can be valid under a free/dynamic schema, or from optional fields), inserts can fail because the table lacks those columns.
- The sink already validates and stores a `schema_config`, but it is not used to define the table schema.

## Severity

- Severity: major
- Priority: P1

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

1. Configure a DatabaseSink with a schema that includes an optional field, e.g. `["id: int", "score: float?"]`.
2. Write a first batch where `score` is absent, e.g. `[{"id": 1}]`.
3. Write a second batch where `score` is present, e.g. `[{"id": 2, "score": 1.0}]`.
4. Observe insert failure (unknown column / compile error) because the table was created without `score`.

## Expected Behavior

- When schema is explicit, the sink should create a table that includes all schema-defined fields (including optional ones), and accept rows that include any subset of them.

## Actual Behavior

- Table columns are derived only from the first row's keys; later valid keys can cause failures.

## Evidence

- `_ensure_table` infers columns from first row keys: `src/elspeth/plugins/sinks/database_sink.py:91-108`
- Sink validates `schema_config` but does not use it for schema creation: `src/elspeth/plugins/sinks/database_sink.py:71-86`
- Writes always call `_ensure_table(rows[0])`, i.e., first-row driven: `src/elspeth/plugins/sinks/database_sink.py:151-158`

## Impact

- User-facing impact: nondeterministic failures depending on row ordering; difficult to operate pipelines reliably.
- Data integrity / security impact: schema configuration is not honored by the sink's side effects; audit trail may not reflect actual DB schema.
- Performance or cost impact: reruns, partial writes, and manual table repair.

## Root Cause Hypothesis

- Implementation uses a "quickstart" inference approach rather than schema-driven schema definition.

## Proposed Fix

- Code changes (modules/files):
  - For explicit schemas, create columns from `schema_config.fields` instead of row keys.
  - Decide how to handle dynamic/free schemas:
    - either enforce a stable column set at run start (collect union from schema or first batch), or
    - support schema evolution (ALTER TABLE) with careful audit/migration policy.
- Config or schema changes:
  - Potentially restrict DatabaseSink to explicit schemas unless schema evolution is implemented.
- Tests to add/update:
  - Add a test for optional field appearing after first batch and assert sink still works.
- Risks or migration steps:
  - Altering existing tables requires migrations; document behavior clearly.

## Architectural Deviations

- Spec or doc reference: `CLAUDE.md` (auditability and reproducibility)
- Observed divergence: declared schema is not used to shape the sink's output schema (table definition).
- Reason (if known): minimal inference implementation.
- Alignment plan or decision needed: define whether sinks are schema-driven or data-driven, and how that affects audit expectations.

## Acceptance Criteria

- With explicit schema config, DatabaseSink creates a table that includes all schema fields and writes batches regardless of which fields appear first.

## Tests

- Suggested tests to run: `pytest tests/plugins/sinks/test_database_sink.py`
- New tests required: yes

## Notes / Links

- Related ticket: `docs/bugs/open/2026-01-19-databasesink-if-exists-replace-ignored.md`

---

## Resolution

**Status:** CLOSED
**Date:** 2026-01-21
**Resolved by:** Claude Opus 4.5

### Root Cause Confirmed

The `_ensure_table` method created table columns from `row.keys()` instead of using the stored `_schema_config.fields` when an explicit schema was configured.

### Fix Applied

Modified `src/elspeth/plugins/sinks/database_sink.py`:

1. Added `SCHEMA_TYPE_TO_SQLALCHEMY` mapping to convert schema field types (str, int, float, bool, any) to SQLAlchemy column types (String, Integer, Float, Boolean).

2. Split `_ensure_table` logic into a new `_create_columns_from_schema_or_row` method:
   - When schema is explicit (not dynamic) and has fields, creates columns from `schema_config.fields` with proper type mapping
   - When schema is dynamic, falls back to inferring from first row keys (original behavior preserved)

3. Updated class docstring to reflect the new behavior.

### Tests Added

Three new tests in `tests/plugins/sinks/test_database_sink.py`:

1. `test_explicit_schema_creates_all_columns_including_optional` - Verifies optional fields present in table even when first row doesn't include them
2. `test_explicit_schema_maps_types_correctly` - Verifies schema types (int, str, float, bool) map to correct SQLAlchemy types
3. `test_dynamic_schema_still_infers_from_row` - Verifies backward compatibility for dynamic schemas

### Verification

- All 15 database sink tests pass
- All 939 plugin tests pass
- Type checking (mypy) passes
- Linting (ruff) passes
