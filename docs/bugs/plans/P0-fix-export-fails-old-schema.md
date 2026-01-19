# Implementation Plan: Schema Compatibility Check for Landscape DB

**Bug:** P0-2026-01-19-export-fails-old-landscape-schema-expand-group-id.md
**Estimated Time:** 1-2 hours
**Complexity:** Low
**Risk:** Low (fail-fast improvement)

## Summary

When using an existing `audit.db` created before `tokens.expand_group_id` was added, the export phase crashes with `sqlite3.OperationalError: no such column`. This plan adds an early schema compatibility check with a clear error message.

## Root Cause

- `LandscapeDB._create_tables()` uses `metadata.create_all()` which only creates missing tables, not missing columns
- When an old DB file exists, it has the old schema without `expand_group_id`
- Export queries `SELECT * FROM tokens` which includes `expand_group_id`, causing SQLite to fail

## Implementation Steps

### Step 1: Add schema validation method to LandscapeDB

**File:** `src/elspeth/core/landscape/database.py`

**Add after `_create_tables()` method:**

```python
# Required columns that have been added since initial schema
# Each entry: (table_name, column_name, added_in_version)
_REQUIRED_COLUMNS: list[tuple[str, str, str]] = [
    ("tokens", "expand_group_id", "0.2.0"),
]


def _validate_schema(self) -> None:
    """Validate that existing database has all required columns.

    Raises:
        SchemaCompatibilityError: If database is missing required columns
    """
    if not self.connection_string.startswith("sqlite"):
        # PostgreSQL handles this differently; skip for now
        return

    from sqlalchemy import inspect

    inspector = inspect(self.engine)

    missing_columns: list[tuple[str, str]] = []

    for table_name, column_name, _version in _REQUIRED_COLUMNS:
        # Check if table exists
        if table_name not in inspector.get_table_names():
            # Table will be created by create_all, skip
            continue

        # Check if column exists
        columns = {c["name"] for c in inspector.get_columns(table_name)}
        if column_name not in columns:
            missing_columns.append((table_name, column_name))

    if missing_columns:
        missing_str = ", ".join(f"{t}.{c}" for t, c in missing_columns)
        raise SchemaCompatibilityError(
            f"Landscape database schema is outdated. "
            f"Missing columns: {missing_str}\n\n"
            f"To fix this, either:\n"
            f"  1. Delete the database file and let ELSPETH recreate it, or\n"
            f"  2. Run: elspeth landscape migrate (when available)\n\n"
            f"Database: {self.connection_string}"
        )
```

### Step 2: Add custom exception

**File:** `src/elspeth/core/landscape/database.py`

**Add near top of file after imports:**

```python
class SchemaCompatibilityError(Exception):
    """Raised when the Landscape database schema is incompatible with current code."""

    pass
```

### Step 3: Call validation during initialization

**Modify `__init__` method:**

```python
def __init__(self, connection_string: str) -> None:
    """Initialize database connection."""
    self.connection_string = connection_string
    self._engine: Engine | None = None
    self._setup_engine()
    self._validate_schema()  # Add this line - check BEFORE create_tables
    self._create_tables()
```

**Also modify `from_url` class method:**

```python
@classmethod
def from_url(cls, url: str, *, create_tables: bool = True) -> Self:
    """Create database from connection URL."""
    engine = create_engine(url, echo=False)
    if url.startswith("sqlite"):
        cls._configure_sqlite(engine)

    instance = cls.__new__(cls)
    instance.connection_string = url
    instance._engine = engine

    # Validate schema before creating tables
    instance._validate_schema()

    if create_tables:
        metadata.create_all(engine)
    return instance
```

### Step 4: Export the exception

**File:** `src/elspeth/core/landscape/__init__.py`

Add to exports:
```python
from elspeth.core.landscape.database import (
    LandscapeDB,
    SchemaCompatibilityError,  # Add this
)

__all__ = [
    ...
    "SchemaCompatibilityError",
    ...
]
```

### Step 5: Add unit tests

**File:** `tests/core/landscape/test_database_schema.py` (new file)

```python
"""Tests for Landscape database schema compatibility checking."""

import tempfile
from pathlib import Path

import pytest
from sqlalchemy import create_engine, text

from elspeth.core.landscape.database import LandscapeDB, SchemaCompatibilityError


class TestSchemaCompatibility:
    """Tests for schema version checking."""

    def test_fresh_database_passes_validation(self, tmp_path: Path):
        """A new database should pass schema validation."""
        db_path = tmp_path / "test.db"
        # Should not raise
        db = LandscapeDB(f"sqlite:///{db_path}")
        db.close()

    def test_in_memory_database_passes_validation(self):
        """In-memory database should pass schema validation."""
        # Should not raise
        db = LandscapeDB.in_memory()
        db.close()

    def test_old_schema_missing_column_fails_validation(self, tmp_path: Path):
        """Database missing required columns should fail with clear error."""
        db_path = tmp_path / "old_schema.db"

        # Create a database with old schema (tokens without expand_group_id)
        old_engine = create_engine(f"sqlite:///{db_path}")
        with old_engine.begin() as conn:
            # Create minimal old schema
            conn.execute(text("""
                CREATE TABLE runs (
                    run_id TEXT PRIMARY KEY,
                    started_at TIMESTAMP NOT NULL
                )
            """))
            conn.execute(text("""
                CREATE TABLE rows (
                    row_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL
                )
            """))
            conn.execute(text("""
                CREATE TABLE tokens (
                    token_id TEXT PRIMARY KEY,
                    row_id TEXT NOT NULL,
                    fork_group_id TEXT,
                    join_group_id TEXT,
                    branch_name TEXT,
                    step_in_pipeline INTEGER,
                    created_at TIMESTAMP NOT NULL
                )
            """))
            # Note: expand_group_id is intentionally missing
        old_engine.dispose()

        # Now try to open with current LandscapeDB
        with pytest.raises(SchemaCompatibilityError) as exc_info:
            LandscapeDB(f"sqlite:///{db_path}")

        error_msg = str(exc_info.value)
        assert "tokens.expand_group_id" in error_msg
        assert "schema is outdated" in error_msg.lower()
        assert "delete the database" in error_msg.lower() or "migrate" in error_msg.lower()

    def test_error_message_includes_remediation(self, tmp_path: Path):
        """Error message should tell user how to fix the problem."""
        db_path = tmp_path / "old_schema.db"

        # Create old schema
        old_engine = create_engine(f"sqlite:///{db_path}")
        with old_engine.begin() as conn:
            conn.execute(text("CREATE TABLE tokens (token_id TEXT PRIMARY KEY)"))
        old_engine.dispose()

        with pytest.raises(SchemaCompatibilityError) as exc_info:
            LandscapeDB(f"sqlite:///{db_path}")

        error_msg = str(exc_info.value)
        # Should include actionable remediation
        assert "Delete" in error_msg or "delete" in error_msg
        assert str(db_path) in error_msg or "sqlite" in error_msg
```

### Step 6: Clean up example database (if committed)

Check if any example `.db` files are tracked in git:

```bash
git ls-files '*.db'
```

If any are found:
- Remove from git: `git rm --cached examples/**/runs/*.db`
- Ensure `.db` is in `.gitignore` (should already be)

### Step 7: Update examples documentation

**File:** `examples/audit_export/README.md` (if exists) or inline comments

Add note:
```markdown
## Troubleshooting

### Schema Compatibility Error

If you see an error like:
> SchemaCompatibilityError: Landscape database schema is outdated

This means you have an old `audit.db` from a previous version. Fix by deleting it:

```bash
rm examples/audit_export/runs/audit.db
```

Then re-run the example.
```

## Testing Checklist

- [ ] Fresh database creation works without error
- [ ] In-memory database works without error
- [ ] Old schema database raises `SchemaCompatibilityError`
- [ ] Error message includes:
  - [ ] Which columns are missing
  - [ ] Clear remediation steps
  - [ ] Database path
- [ ] `from_url()` also validates schema
- [ ] Examples run successfully after deleting old DB

## Run Tests

```bash
# Run new tests
.venv/bin/python -m pytest tests/core/landscape/test_database_schema.py -v

# Run all landscape tests
.venv/bin/python -m pytest tests/core/landscape/ -v

# Test the example
rm -f examples/audit_export/runs/audit.db
.venv/bin/elspeth run -s examples/audit_export/settings.yaml --execute
```

## Acceptance Criteria

1. ✅ Opening an old-schema DB raises `SchemaCompatibilityError` immediately
2. ✅ Error message clearly explains the problem and how to fix it
3. ✅ Fresh databases and in-memory databases work normally
4. ✅ `examples/audit_export` works on a clean checkout
5. ✅ Unit tests cover the validation logic

## Future Work (Not This PR)

- Implement Alembic migrations for schema evolution
- Add `elspeth landscape migrate` command
- Consider auto-migration with `--auto-migrate` flag (with warnings)
- Add schema version tracking in a `_schema_version` table

## Why This Approach

**Alternative considered: Auto-migrate with ALTER TABLE**

Rejected because:
1. Audit data is "Tier 1" - altering it without explicit user consent is risky
2. Migration could fail partway, leaving DB in inconsistent state
3. Different backends (SQLite vs PostgreSQL) have different ALTER semantics
4. Better to fail fast with clear instructions than silently modify legal records

**This approach:**
- Fails immediately with actionable guidance
- No risk of data corruption
- Simple to implement and test
- Buys time for proper Alembic migration infrastructure
