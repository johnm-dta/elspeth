# SQLite PRAGMA Fix Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Ensure all `LandscapeDB` construction paths apply SQLite PRAGMAs (`foreign_keys=ON`, `journal_mode=WAL`) to maintain the "No Orphan Records" audit invariant.

**Architecture:** Extract PRAGMA configuration into a standalone helper that can configure any engine, then call it from all construction paths (`__init__`, `from_url`, `in_memory`).

**Tech Stack:** Python 3.12, SQLAlchemy, pytest

**Bug Reference:** `docs/bugs/open/2026-01-19-sqlite-pragmas-missing-from-url.md`

---

## Root Cause Summary

The factory methods `from_url()` and `in_memory()` use `cls.__new__(cls)` to bypass `__init__()`, which means `_setup_engine()` and `_configure_sqlite()` are never called. The SQLAlchemy event listener for PRAGMAs is never registered on engines created by factory methods.

---

## Scope Discipline

**DO NOT:**
- Add features not in this plan
- Refactor unrelated code
- Change the public API signatures
- Add deprecation warnings or compatibility shims

**DO:**
- Follow TDD exactly as written
- Fix the bug completely in one pass
- Add tests that would have caught this bug

---

## Task 1: Add Failing Tests for PRAGMA Enforcement

**Files:**
- Modify: `tests/core/landscape/test_database.py`

**Step 1: Write the failing tests**

Add to `tests/core/landscape/test_database.py` in the `TestPhase3ADBMethods` class:

```python
def test_in_memory_enables_foreign_keys(self) -> None:
    """Verify in_memory() enables foreign key enforcement."""
    from sqlalchemy import text

    from elspeth.core.landscape.database import LandscapeDB

    db = LandscapeDB.in_memory()
    with db.engine.connect() as conn:
        result = conn.execute(text("PRAGMA foreign_keys"))
        fk_enabled = result.scalar()
        assert fk_enabled == 1, f"Expected foreign_keys=1, got {fk_enabled}"


def test_in_memory_enables_wal_mode(self) -> None:
    """Verify in_memory() sets WAL journal mode.

    Note: In-memory databases report 'memory' for journal_mode,
    which is acceptable since they don't persist to disk.
    """
    from sqlalchemy import text

    from elspeth.core.landscape.database import LandscapeDB

    db = LandscapeDB.in_memory()
    with db.engine.connect() as conn:
        result = conn.execute(text("PRAGMA journal_mode"))
        mode = result.scalar()
        # In-memory DBs report 'memory' not 'wal', which is fine
        assert mode in ("wal", "memory"), f"Expected wal or memory, got {mode}"


def test_from_url_enables_foreign_keys(self, tmp_path: Path) -> None:
    """Verify from_url() enables foreign key enforcement for SQLite."""
    from sqlalchemy import text

    from elspeth.core.landscape.database import LandscapeDB

    db_path = tmp_path / "fk_test.db"
    db = LandscapeDB.from_url(f"sqlite:///{db_path}")
    with db.engine.connect() as conn:
        result = conn.execute(text("PRAGMA foreign_keys"))
        fk_enabled = result.scalar()
        assert fk_enabled == 1, f"Expected foreign_keys=1, got {fk_enabled}"


def test_from_url_enables_wal_mode(self, tmp_path: Path) -> None:
    """Verify from_url() sets WAL journal mode for SQLite."""
    from sqlalchemy import text

    from elspeth.core.landscape.database import LandscapeDB

    db_path = tmp_path / "wal_test.db"
    db = LandscapeDB.from_url(f"sqlite:///{db_path}")
    with db.engine.connect() as conn:
        result = conn.execute(text("PRAGMA journal_mode"))
        mode = result.scalar()
        assert mode == "wal", f"Expected wal, got {mode}"
```

Also add a test to `TestDatabaseConnection` to ensure the existing `__init__` path also tests foreign keys (currently only tests WAL):

```python
def test_sqlite_foreign_keys_enabled(self, tmp_path: Path) -> None:
    """Verify constructor enables foreign key enforcement."""
    from sqlalchemy import text

    from elspeth.core.landscape.database import LandscapeDB

    db_path = tmp_path / "fk_test.db"
    db = LandscapeDB(f"sqlite:///{db_path}")

    with db.engine.connect() as conn:
        result = conn.execute(text("PRAGMA foreign_keys"))
        fk_enabled = result.scalar()
        assert fk_enabled == 1, f"Expected foreign_keys=1, got {fk_enabled}"
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/core/landscape/test_database.py -v`

Expected:
- `test_in_memory_enables_foreign_keys` - FAIL (fk_enabled == 0)
- `test_from_url_enables_foreign_keys` - FAIL (fk_enabled == 0)
- `test_from_url_enables_wal_mode` - FAIL (mode != 'wal')
- `test_sqlite_foreign_keys_enabled` - PASS (uses `__init__`)
- `test_in_memory_enables_wal_mode` - PASS (in-memory reports 'memory')

**Step 3: Commit the failing tests**

```bash
git add tests/core/landscape/test_database.py
git commit -m "test(landscape): add PRAGMA tests for factory constructors

These tests currently fail because from_url() and in_memory() bypass
__init__() and never call _configure_sqlite().

Bug: docs/bugs/open/2026-01-19-sqlite-pragmas-missing-from-url.md"
```

---

## Task 2: Extract PRAGMA Configuration to Static Helper

**Files:**
- Modify: `src/elspeth/core/landscape/database.py`

**Step 1: Refactor `_configure_sqlite` to accept engine parameter**

Replace the existing `_configure_sqlite` method with a static method that can configure any engine:

```python
@staticmethod
def _configure_sqlite(engine: Engine) -> None:
    """Configure SQLite engine for reliability.

    Registers a connection event hook that sets:
    - PRAGMA journal_mode=WAL (better concurrency)
    - PRAGMA foreign_keys=ON (referential integrity)

    Args:
        engine: SQLAlchemy Engine to configure
    """
    @event.listens_for(engine, "connect")
    def set_sqlite_pragma(
        dbapi_connection: object, connection_record: object
    ) -> None:
        cursor = dbapi_connection.cursor()  # type: ignore[attr-defined]
        # Enable WAL mode for better concurrency
        cursor.execute("PRAGMA journal_mode=WAL")
        # Enable foreign key enforcement
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()
```

**Step 2: Update `_setup_engine` to use the static method**

```python
def _setup_engine(self) -> None:
    """Create and configure the database engine."""
    self._engine = create_engine(
        self.connection_string,
        echo=False,  # Set True for SQL debugging
    )

    # SQLite-specific configuration
    if self.connection_string.startswith("sqlite"):
        LandscapeDB._configure_sqlite(self._engine)
```

**Step 3: Run existing tests to verify no regression**

Run: `pytest tests/core/landscape/test_database.py::TestDatabaseConnection -v`

Expected: All PASS (refactor is behavior-preserving for `__init__` path)

**Step 4: Commit the refactor**

```bash
git add src/elspeth/core/landscape/database.py
git commit -m "refactor(landscape): extract _configure_sqlite as static method

Prepares for reuse in factory constructors.
No behavior change for __init__ path."
```

---

## Task 3: Fix `in_memory()` Factory Method

**Files:**
- Modify: `src/elspeth/core/landscape/database.py`

**Step 1: Add PRAGMA configuration to `in_memory()`**

Update the `in_memory` classmethod:

```python
@classmethod
def in_memory(cls) -> Self:
    """Create an in-memory SQLite database for testing.

    Tables are created automatically.

    Returns:
        LandscapeDB instance with in-memory SQLite
    """
    engine = create_engine("sqlite:///:memory:", echo=False)
    cls._configure_sqlite(engine)  # <-- ADD THIS LINE
    metadata.create_all(engine)
    instance = cls.__new__(cls)
    instance.connection_string = "sqlite:///:memory:"
    instance._engine = engine
    return instance
```

**Step 2: Run tests to verify in_memory tests pass**

Run: `pytest tests/core/landscape/test_database.py::TestPhase3ADBMethods::test_in_memory_enables_foreign_keys -v`

Expected: PASS

Run: `pytest tests/core/landscape/test_database.py::TestPhase3ADBMethods::test_in_memory_enables_wal_mode -v`

Expected: PASS

**Step 3: Commit the fix**

```bash
git add src/elspeth/core/landscape/database.py
git commit -m "fix(landscape): apply SQLite PRAGMAs in in_memory() factory

Calls _configure_sqlite() to enable foreign_keys=ON and journal_mode=WAL.

Bug: docs/bugs/open/2026-01-19-sqlite-pragmas-missing-from-url.md"
```

---

## Task 4: Fix `from_url()` Factory Method

**Files:**
- Modify: `src/elspeth/core/landscape/database.py`

**Step 1: Add PRAGMA configuration to `from_url()`**

Update the `from_url` classmethod:

```python
@classmethod
def from_url(cls, url: str, *, create_tables: bool = True) -> Self:
    """Create database from connection URL.

    Args:
        url: SQLAlchemy connection URL
        create_tables: Whether to create tables if they don't exist.
                       Set to False when connecting to an existing database.

    Returns:
        LandscapeDB instance
    """
    engine = create_engine(url, echo=False)
    # SQLite-specific configuration
    if url.startswith("sqlite"):
        cls._configure_sqlite(engine)
    if create_tables:
        metadata.create_all(engine)
    instance = cls.__new__(cls)
    instance.connection_string = url
    instance._engine = engine
    return instance
```

**Step 2: Run tests to verify from_url tests pass**

Run: `pytest tests/core/landscape/test_database.py::TestPhase3ADBMethods::test_from_url_enables_foreign_keys -v`

Expected: PASS

Run: `pytest tests/core/landscape/test_database.py::TestPhase3ADBMethods::test_from_url_enables_wal_mode -v`

Expected: PASS

**Step 3: Run full test suite to verify no regressions**

Run: `pytest tests/core/landscape/test_database.py -v`

Expected: All tests PASS

**Step 4: Commit the fix**

```bash
git add src/elspeth/core/landscape/database.py
git commit -m "fix(landscape): apply SQLite PRAGMAs in from_url() factory

Calls _configure_sqlite() for SQLite URLs to enable foreign_keys=ON
and journal_mode=WAL.

Bug: docs/bugs/open/2026-01-19-sqlite-pragmas-missing-from-url.md"
```

---

## Task 5: Run Full Test Suite and Close Bug

**Step 1: Run full landscape tests**

Run: `pytest tests/core/landscape/ -v`

Expected: All tests PASS

**Step 2: Run CLI tests (uses from_url)**

Run: `pytest tests/cli/ -v`

Expected: All tests PASS

**Step 3: Move bug report to closed**

```bash
mkdir -p docs/bugs/closed
git mv docs/bugs/open/2026-01-19-sqlite-pragmas-missing-from-url.md docs/bugs/closed/
```

**Step 4: Add resolution notes to bug report**

Append to the moved bug report:

```markdown
## Resolution

**Fixed in commits:**
- `test(landscape): add PRAGMA tests for factory constructors`
- `refactor(landscape): extract _configure_sqlite as static method`
- `fix(landscape): apply SQLite PRAGMAs in in_memory() factory`
- `fix(landscape): apply SQLite PRAGMAs in from_url() factory`

**Root cause:** Factory methods used `cls.__new__(cls)` to bypass `__init__()` without calling `_configure_sqlite()`.

**Fix:** Made `_configure_sqlite()` a static method and called it from all construction paths.

**Closed:** 2026-01-20
```

**Step 5: Commit the closure**

```bash
git add docs/bugs/
git commit -m "docs(bugs): close sqlite-pragmas-missing-from-url

All construction paths now apply SQLite PRAGMAs correctly.
Tests verify foreign_keys=ON and journal_mode=WAL for:
- LandscapeDB.__init__()
- LandscapeDB.from_url()
- LandscapeDB.in_memory()"
```

---

## Verification Checklist

Before considering this complete:

- [ ] `pytest tests/core/landscape/test_database.py -v` - All PASS
- [ ] `pytest tests/cli/ -v` - All PASS (CLI uses from_url)
- [ ] `pytest tests/ -v` - Full suite PASS (many tests use in_memory)
- [ ] Bug report moved to `docs/bugs/closed/`
