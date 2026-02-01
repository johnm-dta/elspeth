# Bug Report: Schema validation accepts non-composite FK to nodes for error tables

## Summary

- `_REQUIRED_FOREIGN_KEYS` only checks single-column FKs, not the composite `(node_id, run_id)` requirement for nodes table references.

## Severity

- Severity: moderate
- Priority: P2

## Reporter

- Name or handle: Codex (triaged by Claude)
- Date: 2026-01-31

## Evidence

- `src/elspeth/core/landscape/database.py:43-50` - `_REQUIRED_FOREIGN_KEYS` only lists single-column FKs.
- Nodes table has composite PK `(node_id, run_id)` per CLAUDE.md
- Validation doesn't enforce composite FK requirement

## Impact

- User-facing impact: Stale SQLite databases could have invalid schema
- Data integrity: FK validation incomplete

## Proposed Fix

- Add composite FK validation for tables referencing nodes

## Acceptance Criteria

- Schema validation detects missing composite FK to nodes table

## Verification (2026-02-01)

**Status: STILL VALID**

- Schema validation still checks only single-column FKs and does not enforce composite `(node_id, run_id)` references. (`src/elspeth/core/landscape/database.py:43-50`)

---

## Resolution (2026-02-02)

**Status: CLOSED - NOT A BUG**

### Analysis

The bug report assumes a scenario that **cannot occur in practice**:

1. **Zero users** - No one has old databases from the transitional period
2. **No legacy code policy** - Old DB formats are explicitly not supported; the error message instructs users to delete the database
3. **SQLite enforces integrity anyway** - Even if validation passed a single-column FK to a composite PK table, SQLite would reject ALL inserts with "foreign key mismatch"

### Verification

```python
# SQLite rejects inserts when single-column FK references composite PK table:
>>> INSERT INTO errors_single (error_id, node_id) VALUES ('err1', 'node1')
sqlite3.OperationalError: foreign key mismatch - "errors_single" referencing "nodes"
```

**Data corruption is impossible** - SQLite's FK enforcement is the actual safety layer, not this validation.

### Why This Validation Exists

The `_validate_schema()` check is a **developer convenience** to catch stale local `audit.db` files during development. It was never intended as a production safety mechanism.

Per the error message:
> "To fix this, either:
>   1. Delete the database file and let ELSPETH recreate it, or
>   2. Run: elspeth landscape migrate (when available)"

The expected action is to delete the database, not to preserve/migrate old formats.

### Conclusion

| Scenario | Can It Happen? | Impact |
|----------|---------------|--------|
| Developer with stale local DB | Yes (development) | Told to delete, no data loss |
| User with transitional-period DB | No (zero users) | N/A |
| Validation passes, insert fails | Theoretical only | Confusing error, not corruption |

This is a theoretical edge case in a validation convenience feature, not a data integrity bug.

### Closed By

- Reviewer: Claude Opus 4.5
- Date: 2026-02-02
- Method: Systematic debugging - verified SQLite FK enforcement prevents the claimed impact
