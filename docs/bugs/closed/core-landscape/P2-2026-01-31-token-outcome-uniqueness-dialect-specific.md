# Bug Report: token_outcomes terminal uniqueness only enforced on SQLite/Postgres

## Summary

- The partial unique index for terminal token outcomes uses dialect-specific clauses (`sqlite_where`, `postgresql_where`). Other backends won't enforce terminal uniqueness.

## Severity

- Severity: moderate
- Priority: P2

## Reporter

- Name or handle: Codex (triaged by Claude)
- Date: 2026-01-31

## Evidence

- `src/elspeth/core/landscape/schema.py:157-165` - partial unique index uses dialect-specific clauses (`sqlite_where`, `postgresql_where`).
- Only SQLite and PostgreSQL get the constraint
- MySQL and other backends have no enforcement

## Impact

- User-facing impact: Could have duplicate terminal outcomes on non-SQLite/Postgres
- Data integrity: Uniqueness constraint not portable

## Proposed Fix

- Document SQLite/Postgres as only supported backends, or add application-level uniqueness check

## Acceptance Criteria

- Terminal uniqueness enforced on all supported backends

## Verification (2026-02-01)

**Status: STILL VALID**

- The terminal-uniqueness constraint remains dialect-specific to SQLite/Postgres. (`src/elspeth/core/landscape/schema.py:157-165`)

---

## Resolution (2026-02-02)

**Status: CLOSED - INVALID / NOT A BUG**

### Root Cause Analysis

This bug report is based on a **false premise**. ELSPETH only supports SQLite and PostgreSQL backends, and this is enforced at multiple levels:

1. **Type-level enforcement** (`src/elspeth/core/config.py:488`):
   ```python
   backend: Literal["sqlite", "postgresql"] = Field(...)
   ```

2. **Pydantic validation**: Any attempt to use `mysql`, `oracle`, `mariadb`, or other backends is **rejected at configuration load time** before any database code runs.

3. **Documentation**: Both `database.py` docstring and `configuration.md` explicitly state only SQLite and PostgreSQL are supported.

### Verification

```python
# Pydantic correctly rejects unsupported backends:
>>> LandscapeSettings(backend="mysql")
ValidationError: Input should be 'sqlite' or 'postgresql'
```

### Why This Is Not A Bug

| Bug Report Claim | Reality |
|------------------|---------|
| "Other backends won't enforce terminal uniqueness" | Other backends cannot be used - Pydantic rejects them |
| "MySQL and other backends have no enforcement" | MySQL cannot be configured as a backend |
| Impact: "Could have duplicate terminal outcomes" | Impossible - config validation prevents this |

The partial unique index with `sqlite_where` and `postgresql_where` covers **100% of officially supported backends**.

### Proposed Fix Status

The proposed fix ("Document SQLite/Postgres as only supported backends") is **already implemented**:
- `config.py` enforces it at the type level
- `database.py` docstring states it (line 1-6)
- `configuration.md` documents it (line 464)

### Closed By

- Reviewer: Claude Opus 4.5
- Date: 2026-02-02
- Method: Systematic debugging protocol - traced configuration enforcement path
