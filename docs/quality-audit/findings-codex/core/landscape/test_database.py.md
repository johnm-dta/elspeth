Using using-quality-engineering (mutation-testing) to assess test effectiveness and assertion strength.

# Test Defect Report

## Summary

- Table-creation tests only verify a small subset of tables (or just absence of `runs`), so partial schema creation regressions can pass unnoticed.

## Severity

- Severity: minor
- Priority: P2

## Category

- Weak Assertions

## Evidence

- `tests/core/landscape/test_database.py:20`
```python
tables = inspector.get_table_names()
assert "runs" in tables
assert "nodes" in tables
```
- `tests/core/landscape/test_database.py:72` and `tests/core/landscape/test_database.py:93` only check for `"runs"` on in-memory/from_url DBs; no assertion for the full schema.
- `tests/core/landscape/test_database.py:109`
```python
assert "runs" not in inspector.get_table_names()  # No tables!
```
- `src/elspeth/core/landscape/database.py:89` uses `metadata.create_all(self.engine)`, which implies all tables should be present, not just two.

## Impact

- A regression that creates only a subset of tables (e.g., missing `token_outcomes` or `artifacts`) would still pass these tests.
- This creates false confidence that `LandscapeDB` initializes the full audit schema.
- Downstream tests may fail later with harder-to-diagnose errors.

## Root Cause Hypothesis

- Tests were added when only core tables mattered and not updated as the schema expanded.
- Reliance on other schema tests led to minimal assertions in this file.

## Recommended Fix

- Strengthen table creation assertions to compare against `metadata.tables` and ensure the full expected schema is created.
- Example:
```python
from elspeth.core.landscape.schema import metadata

expected = set(metadata.tables.keys())
tables = set(inspector.get_table_names())
assert expected.issubset(tables)
```
- For `create_tables=False`, assert the table list is empty rather than only checking `"runs"` is missing.
- Priority justification: Prevents partial-schema regressions from slipping through the connection-management test suite.
---
# Test Defect Report

## Summary

- Schema compatibility tests cover missing columns but do not test missing required foreign keys enforced by `_validate_schema`.

## Severity

- Severity: major
- Priority: P1

## Category

- Missing Tier 1 Corruption Tests

## Evidence

- `src/elspeth/core/landscape/database.py:30` defines required foreign keys, and `src/elspeth/core/landscape/database.py:123` enforces them:
```python
_REQUIRED_FOREIGN_KEYS = [
    ("validation_errors", "node_id", "nodes"),
    ("transform_errors", "token_id", "tokens"),
    ("transform_errors", "transform_id", "nodes"),
]
...
for table_name, column_name, referenced_table in _REQUIRED_FOREIGN_KEYS:
    ...
    if not has_correct_fk:
        missing_fks.append((table_name, column_name, referenced_table))
```
- `tests/core/landscape/test_database.py:187` and `tests/core/landscape/test_database.py:244` only validate missing columns (e.g., `tokens.expand_group_id`) and never construct a schema missing required FKs.

## Impact

- A regression that removes or breaks FK validation would not be detected, allowing Tier 1 audit tables with missing FK constraints to pass validation.
- This risks orphan error records and undermines audit integrity.

## Root Cause Hypothesis

- FK validation was added after the original column-compatibility tests, but no tests were added for the new FK checks.

## Recommended Fix

- Add a test that creates an SQLite database with the required tables but intentionally omits FK constraints (e.g., `validation_errors.node_id` or `transform_errors.token_id`) and assert `SchemaCompatibilityError`.
- Example:
```python
with old_engine.begin() as conn:
    conn.execute(text("CREATE TABLE nodes (node_id TEXT PRIMARY KEY)"))
    conn.execute(text("CREATE TABLE validation_errors (error_id TEXT PRIMARY KEY, node_id TEXT NOT NULL)"))  # no FK
with pytest.raises(SchemaCompatibilityError) as exc_info:
    LandscapeDB(f"sqlite:///{db_path}")
assert "Missing foreign keys" in str(exc_info.value)
```
- Priority justification: FK enforcement is Tier 1 audit integrity; missing tests here risk silent acceptance of corrupted audit schemas.
