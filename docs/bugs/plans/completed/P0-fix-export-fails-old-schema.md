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

### Step 1: Add schema validation constant and method to LandscapeDB

**File:** `src/elspeth/core/landscape/database.py`

**Add at module level (after imports, before class definition, ~line 16):**

```python
# Required columns that have been added since initial schema.
# Used by _validate_schema() to detect outdated SQLite databases.
_REQUIRED_COLUMNS: list[tuple[str, str]] = [
    ("tokens", "expand_group_id"),
]
```

**Add as instance method inside `LandscapeDB` class (after `_create_tables()`, ~line 71):**

```python
def _validate_schema(self) -> None:
    """Validate that existing database has all required columns.

    Only validates SQLite databases. PostgreSQL deployments are expected
    to use Alembic migrations which handle schema evolution properly.
    This check catches developers using stale local audit.db files.

    Raises:
        SchemaCompatibilityError: If database is missing required columns
    """
    if not self.connection_string.startswith("sqlite"):
        return

    from sqlalchemy import inspect

    inspector = inspect(self.engine)

    missing_columns: list[tuple[str, str]] = []

    for table_name, column_name in _REQUIRED_COLUMNS:
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

**Modify `__init__` method (~line 21):**

```python
def __init__(self, connection_string: str) -> None:
    """Initialize database connection."""
    self.connection_string = connection_string
    self._engine: Engine | None = None
    self._setup_engine()
    self._validate_schema()  # Add this line - check BEFORE create_tables
    self._create_tables()
```

**Also modify `from_url` class method (~line 113):**

```python
@classmethod
def from_url(cls, url: str, *, create_tables: bool = True) -> Self:
    """Create database from connection URL."""
    engine = create_engine(url, echo=False)
    if url.startswith("sqlite"):
        cls._configure_sqlite(engine)

    # Create instance first - _validate_schema needs self.connection_string and self.engine
    instance = cls.__new__(cls)
    instance.connection_string = url
    instance._engine = engine

    # Validate BEFORE create_all - catches old schema with missing columns
    # before we try to use it. For fresh DBs, validation passes (no tables yet).
    instance._validate_schema()

    if create_tables:
        metadata.create_all(engine)
    return instance
```

### Step 3a: Note on `in_memory()` - No changes needed

The `in_memory()` factory method bypasses `__init__` and creates a fresh in-memory
database with `metadata.create_all(engine)`. Since:

1. In-memory databases are always fresh (no pre-existing schema)
2. `create_all()` uses the current schema definition
3. There's nothing to validate

No validation call is needed. The method remains unchanged.

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

**File:** `tests/core/landscape/test_database.py` (add to existing file)

Add a new test class at the end of the file:

```python
class TestSchemaCompatibility:
    """Tests for schema version checking."""

    def test_fresh_database_passes_validation(self, tmp_path: Path):
        """A new database should pass schema validation."""
        from elspeth.core.landscape.database import LandscapeDB

        db_path = tmp_path / "test.db"
        # Should not raise
        db = LandscapeDB(f"sqlite:///{db_path}")
        db.close()

    def test_in_memory_database_passes_validation(self):
        """In-memory database should pass schema validation."""
        from elspeth.core.landscape.database import LandscapeDB

        # Should not raise
        db = LandscapeDB.in_memory()
        db.close()

    def test_old_schema_missing_column_fails_validation(self, tmp_path: Path):
        """Database missing required columns should fail with clear error."""
        from sqlalchemy import create_engine, text

        from elspeth.core.landscape.database import LandscapeDB, SchemaCompatibilityError

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

    def test_from_url_with_old_schema_fails(self, tmp_path: Path):
        """Verify from_url() catches old schema before create_all()."""
        from sqlalchemy import create_engine, text

        from elspeth.core.landscape.database import LandscapeDB, SchemaCompatibilityError

        db_path = tmp_path / "old_schema.db"

        # Create old schema
        old_engine = create_engine(f"sqlite:///{db_path}")
        with old_engine.begin() as conn:
            conn.execute(text("""
                CREATE TABLE tokens (
                    token_id TEXT PRIMARY KEY,
                    row_id TEXT NOT NULL,
                    created_at TIMESTAMP NOT NULL
                )
            """))
        old_engine.dispose()

        # from_url should fail even with create_tables=True
        with pytest.raises(SchemaCompatibilityError) as exc_info:
            LandscapeDB.from_url(f"sqlite:///{db_path}", create_tables=True)

        assert "expand_group_id" in str(exc_info.value)

    def test_error_message_includes_remediation(self, tmp_path: Path):
        """Error message should tell user how to fix the problem."""
        from sqlalchemy import create_engine, text

        from elspeth.core.landscape.database import LandscapeDB, SchemaCompatibilityError

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
- [ ] Old schema database via `__init__` raises `SchemaCompatibilityError`
- [ ] Old schema database via `from_url()` raises `SchemaCompatibilityError`
- [ ] Error message includes:
  - [ ] Which columns are missing
  - [ ] Clear remediation steps
  - [ ] Database path
- [ ] Examples run successfully after deleting old DB

## Run Tests

```bash
# Run schema compatibility tests
.venv/bin/python -m pytest tests/core/landscape/test_database.py::TestSchemaCompatibility -v

# Run all landscape tests
.venv/bin/python -m pytest tests/core/landscape/ -v

# Test the example
rm -f examples/audit_export/runs/audit.db
.venv/bin/elspeth run -s examples/audit_export/settings.yaml --execute
```

## Acceptance Criteria

1. ✅ Opening an old-schema DB via `__init__` raises `SchemaCompatibilityError` immediately
2. ✅ Opening an old-schema DB via `from_url()` raises `SchemaCompatibilityError` immediately
3. ✅ Error message clearly explains the problem and how to fix it
4. ✅ Fresh databases and in-memory databases work normally
5. ✅ `examples/audit_export` works on a clean checkout
6. ✅ Unit tests cover all entry points (`__init__`, `from_url`, `in_memory`)

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
