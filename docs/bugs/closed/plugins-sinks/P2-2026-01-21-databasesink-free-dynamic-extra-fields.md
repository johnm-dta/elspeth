# Bug Report: DatabaseSink rejects extra fields in free/dynamic schemas

## FIXED: 2026-02-02

**Status:** FIXED (Option B implemented)

**Fix:** DatabaseSink now fails fast during `__init__` if `schema_config.allows_extra_fields` is True, with a clear error message directing users to JSONSink for flexible schemas.

**Location:** `src/elspeth/plugins/sinks/database_sink.py` lines 109-115

**Tests:** `tests/plugins/sinks/test_database_sink.py::TestDatabaseSinkSchemaValidation` (3 tests)

---

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

---

## VERIFICATION: 2026-02-01

**Status:** STILL VALID

- Table columns are still created from explicit schema fields (free mode), so extras are not represented. (`src/elspeth/plugins/sinks/database_sink.py:242-261`)
- Inserts still pass full row dicts, so extra fields cause SQLAlchemy errors when schema is explicit. (`src/elspeth/plugins/sinks/database_sink.py:311-315`)

## VERIFICATION: 2026-01-25

**Status:** STILL VALID

**Verified By:** Claude Code P2 verification wave 6b

**Current Code Analysis:**

Examined `/home/john/elspeth-rapid/src/elspeth/plugins/sinks/database_sink.py` (current HEAD on branch `fix/rc1-bug-burndown-session-4`). The bug is still present:

1. **Schema Configuration**: `SchemaConfig.allows_extra_fields` returns `True` for both `mode="free"` and dynamic schemas (line 269 in `src/elspeth/contracts/schema.py`).

2. **Column Creation Logic** (lines 176-195 in `database_sink.py`):
   - For explicit schemas (including `mode="free"`), columns are created ONLY from `self._schema_config.fields`
   - For dynamic schemas, columns are inferred from first row keys
   - **Critical issue**: Free mode schemas are explicit (not dynamic), so they take the explicit path and create columns only from defined fields

3. **Insert Logic** (line 243): `conn.execute(insert(self._table), rows)` attempts to insert ALL fields from the row dict, not just columns that exist in the table.

4. **Failure Point**: When a free schema defines `["id: int"]` but receives rows like `{"id": 1, "extra": "x"}`, the table has only an `id` column, but SQLAlchemy's `insert()` receives a dict with both `id` and `extra`, causing an `ArgumentError` for unconsumed column names.

**Example Failure Case:**
```python
# Schema allows extras
config = {"mode": "free", "fields": ["id: int"]}
# Row has extra field
row = {"id": 1, "extra_field": "data"}
# Table created with only: Column("id", Integer)
# Insert fails: ArgumentError: Unconsumed column names: extra_field
```

**Git History:**

Checked commits since 2026-01-21:
- `57c57f5` (Jan 21): Fixed 8 RC1 bugs, including DatabaseSink dialect-safe Table.drop() for replace mode - **did NOT address extra fields**
- `7ee7c51`: Added self-validation to all builtin plugins - **did NOT address extra fields**
- No commits specifically addressing free/dynamic schema extra field handling

**Test Coverage Gap:**

The test `test_explicit_schema_creates_all_columns_including_optional` (line 184) uses `mode="free"` but only tests **optional fields that ARE defined in the schema** (e.g., `"score: float?"`), not truly extra fields that are NOT in the schema definition. No test exists for the reported bug scenario.

**Root Cause Confirmed:**

Yes. The architectural mismatch is clear:

1. `SchemaConfig` allows extra fields for free/dynamic modes (contract layer says "yes")
2. `DatabaseSink` creates fixed table schema from defined fields only (implementation layer says "no columns for extras")
3. SQLAlchemy insert receives full row dicts, fails when dict keys don't match table columns

This is a **contract violation** between the schema system and the sink implementation.

**Recommendation:**

**Keep open** - This is a valid architectural bug requiring a design decision:

**Option A**: Strip extra fields before insert (silently discard them)
**Option B**: Fail fast during plugin initialization if `allows_extra_fields` is True (reject free/dynamic schemas)
**Option C**: Dynamically add columns on first encounter of new fields (schema evolution)

Each option has tradeoffs:
- **A** silently loses data (violates audit principles)
- **B** restricts DatabaseSink to strict-only schemas (breaks free/dynamic use cases)
- **C** requires table alteration (migration complexity, potential race conditions)

Recommended approach: **Option B** with clear error message. DatabaseSink should reject free/dynamic schemas in `_validate_self_consistency()` because relational databases inherently require fixed schemas. Users needing flexible schemas should use JSONSink or other schema-less sinks.

This maintains audit integrity (no silent data loss) and provides clear guidance to users.
