# Tier 1 Audit Integrity Hardening — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the remaining Tier 1 integrity gaps: missing NaN/Infinity rejection in 6 audit serialization paths, and zero test coverage for the `_database_ops.py` write-guard module.

**Architecture:** Two independent concerns: (1) add `allow_nan=False` to all `json.dumps()` calls in `data_flow_repository.py` that write to the audit trail, with tests proving NaN/Infinity are rejected; (2) add dedicated unit tests for `_database_ops.py` covering both happy-path operations and the critical `rowcount == 0` crash behavior.

**Tech Stack:** Python 3.12, pytest, SQLAlchemy Core (in-memory SQLite for `_database_ops` tests)

---

## File Map

### New Files
- `tests/unit/core/landscape/test_database_ops.py` — dedicated tests for `_database_ops.py`
- `tests/unit/core/landscape/test_data_flow_nan_rejection.py` — NaN/Infinity rejection tests for `data_flow_repository.py`

### Modified Files
- `src/elspeth/core/landscape/data_flow_repository.py:336,527,740,1299,1384,1406` — add `allow_nan=False`

---

## Task 1: Add `allow_nan=False` to all audit-path `json.dumps` calls

**Rationale:** The Data Manifesto requires NaN/Infinity rejection at all Tier 1 boundaries. Six `json.dumps()` calls in `data_flow_repository.py` are missing `allow_nan=False`. Every other Tier 1 serialization site (`journal.py:190`, `formatters.py:129`, `checkpoint/serialization.py:156`) already passes it. This is a consistency bug.

**Files:**
- Modify: `src/elspeth/core/landscape/data_flow_repository.py:336,527,740,1299,1384,1406`
- Create: `tests/unit/core/landscape/test_data_flow_nan_rejection.py`

- [ ] **Step 1: Write failing tests**

These tests need a real `DataFlowRepository` instance with a `LandscapeDB` to exercise the actual `json.dumps` paths. The test should verify that passing data containing `float("nan")` or `float("inf")` raises `ValueError` (which is what `json.dumps(..., allow_nan=False)` raises).

The tests for this task take two forms:

1. **AST audit test** — statically verify that every `json.dumps` call in the file passes `allow_nan=False`. This is the primary guard against regressions.
2. **Behavioral guard tests** — verify `json.dumps(allow_nan=False)` actually rejects NaN/Infinity (belt-and-suspenders).

We do NOT try to reach each `json.dumps` call through the full `DataFlowRepository` method stack — those paths require extensive audit trail state setup and are already covered by integration tests. Instead, the AST test proves the flag is present, and the behavioral tests prove the flag works.

**⚠️ Reviewer note (I1):** `DataFlowRepository` has a 7-parameter constructor — do NOT try to instantiate it directly in these tests. The AST approach avoids this entirely.

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/unit/core/landscape/test_data_flow_nan_rejection.py -v`
Expected: Tests should fail or be incomplete skeletons

- [ ] **Step 3: Add `allow_nan=False` to all 6 sites**

In `src/elspeth/core/landscape/data_flow_repository.py`:

**Line 336** (quarantine fallback):
```python
# Before:
payload_bytes = json.dumps({"_repr": repr(data)}).encode("utf-8")
# After:
payload_bytes = json.dumps({"_repr": repr(data)}, allow_nan=False).encode("utf-8")
```

**Line 527** (fork token branches):
```python
# Before:
expected_branches_json=json.dumps(branches),
# After:
expected_branches_json=json.dumps(branches, allow_nan=False),
```

**Line 740** (expand token count):
```python
# Before:
expected_branches_json=json.dumps({"count": count}),
# After:
expected_branches_json=json.dumps({"count": count}, allow_nan=False),
```

**Line 1299** (validation error metadata):
```python
# Before:
row_data_json = json.dumps(metadata.to_dict())
# After:
row_data_json = json.dumps(metadata.to_dict(), allow_nan=False)
```

**Line 1384** (transform error details fallback):
```python
# Before:
error_details_json = json.dumps(
    {
        "__non_canonical__": True,
        "repr": repr(error_details)[:500],
# After:
error_details_json = json.dumps(
    {
        "__non_canonical__": True,
        "repr": repr(error_details)[:500],
```
(add `allow_nan=False` as a keyword arg to this `json.dumps` call)

**Line 1406** (transform error metadata):
```python
# Before:
row_data_json = json.dumps(metadata.to_dict())
# After:
row_data_json = json.dumps(metadata.to_dict(), allow_nan=False)
```

- [ ] **Step 4: Complete the tests**

Now that `allow_nan=False` is in place, write targeted tests that exercise the serialization. The simplest approach is to test at the `json.dumps` level:

```python
import json
import math
import pytest


class TestAllowNanFalseGuard:
    """Verify json.dumps(allow_nan=False) rejects NaN and Infinity.

    This is a belt-and-suspenders test: we verify the behavior of the
    flag itself, then the integration tests verify the production paths
    don't regress.
    """

    def test_nan_rejected(self) -> None:
        with pytest.raises(ValueError, match="Out of range float values"):
            json.dumps({"value": float("nan")}, allow_nan=False)

    def test_infinity_rejected(self) -> None:
        with pytest.raises(ValueError, match="Out of range float values"):
            json.dumps({"value": float("inf")}, allow_nan=False)

    def test_neg_infinity_rejected(self) -> None:
        with pytest.raises(ValueError, match="Out of range float values"):
            json.dumps({"value": float("-inf")}, allow_nan=False)

    def test_normal_float_passes(self) -> None:
        result = json.dumps({"value": 3.14}, allow_nan=False)
        assert "3.14" in result
```

Additionally, use `grep` to verify there are zero unguarded `json.dumps` calls remaining in the file:

```python
class TestNoUnguardedJsonDumps:
    """Audit: every json.dumps in data_flow_repository.py must pass allow_nan=False."""

    def test_all_json_dumps_have_allow_nan_false(self) -> None:
        import ast
        from pathlib import Path

        import elspeth.core.landscape.data_flow_repository as mod

        source = Path(mod.__file__).read_text()
        tree = ast.parse(source)

        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "dumps"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "json"
            ):
                kwarg_names = [kw.arg for kw in node.keywords]
                assert "allow_nan" in kwarg_names, (
                    f"json.dumps at line {node.lineno} is missing allow_nan=False"
                )
```

- [ ] **Step 5: Run tests**

Run: `.venv/bin/python -m pytest tests/unit/core/landscape/test_data_flow_nan_rejection.py -v`
Expected: ALL PASS

Run: `.venv/bin/python -m pytest tests/ -k "data_flow" -v`
Expected: ALL PASS (existing tests unaffected)

- [ ] **Step 6: Commit**

```bash
git add src/elspeth/core/landscape/data_flow_repository.py tests/unit/core/landscape/test_data_flow_nan_rejection.py
git commit -m "fix: add allow_nan=False to 6 audit-path json.dumps calls in data_flow_repository"
```

---

## Task 2: Add dedicated tests for `_database_ops.py`

**Rationale:** `_database_ops.py` (71 lines) is the single enforcement point for "every audit write must succeed." Its `execute_insert` and `execute_update` methods crash with `AuditIntegrityError` when `rowcount == 0`. This critical behavior has zero dedicated tests — it's only tested indirectly through higher-level integration tests.

**Files:**
- Create: `tests/unit/core/landscape/test_database_ops.py`

- [ ] **Step 1: Write the test file**

The tests need a real in-memory SQLite database with a simple table to exercise actual SQL execution. Don't mock SQLAlchemy — these tests must verify real database behavior.

```python
# tests/unit/core/landscape/test_database_ops.py
"""Dedicated tests for _database_ops.py — Tier 1 write guard.

Tests verify:
1. Happy path: insert/update/fetchone/fetchall work correctly
2. Critical path: rowcount == 0 raises AuditIntegrityError
3. Context strings appear in error messages
"""

import pytest
import sqlalchemy as sa

from elspeth.contracts.errors import AuditIntegrityError
from elspeth.core.landscape._database_ops import DatabaseOps
from elspeth.core.landscape.database import LandscapeDB


@pytest.fixture
def db() -> LandscapeDB:
    """In-memory SQLite database with a simple test table."""
    ldb = LandscapeDB.in_memory()
    # Create a minimal test table (separate from Landscape schema)
    metadata = sa.MetaData()
    test_table = sa.Table(
        "test_rows",
        metadata,
        sa.Column("id", sa.String, primary_key=True),
        sa.Column("value", sa.String),
    )
    with ldb.connection() as conn:
        metadata.create_all(conn.engine)
    return ldb


@pytest.fixture
def ops(db: LandscapeDB) -> DatabaseOps:
    return DatabaseOps(db)


@pytest.fixture
def test_table() -> sa.Table:
    metadata = sa.MetaData()
    return sa.Table(
        "test_rows",
        metadata,
        sa.Column("id", sa.String, primary_key=True),
        sa.Column("value", sa.String),
    )


class TestExecuteInsert:
    def test_insert_succeeds(self, ops: DatabaseOps, test_table: sa.Table) -> None:
        """Normal insert with rowcount > 0 does not raise."""
        ops.execute_insert(
            test_table.insert().values(id="row1", value="hello"),
            context="test insert",
        )

    def test_insert_zero_rows_raises_audit_integrity_error(
        self, ops: DatabaseOps, test_table: sa.Table
    ) -> None:
        """Insert that affects zero rows must crash with AuditIntegrityError.

        Use INSERT OR IGNORE via sa.text() to get rowcount=0 without SQL error.
        Must go through ops.execute_insert() to test the actual guard.
        """
        # First insert succeeds
        ops.execute_insert(
            test_table.insert().values(id="row1", value="first"),
        )
        # INSERT OR IGNORE with duplicate PK → rowcount=0, no SQL error
        with pytest.raises(AuditIntegrityError, match="zero rows affected"):
            ops.execute_insert(
                sa.text("INSERT OR IGNORE INTO test_rows (id, value) VALUES ('row1', 'dup')"),
            )

    def test_insert_context_in_error_message(
        self, ops: DatabaseOps, test_table: sa.Table
    ) -> None:
        """Context string appears in the AuditIntegrityError message."""
        ops.execute_insert(
            test_table.insert().values(id="row1", value="first"),
        )
        with pytest.raises(AuditIntegrityError, match="my context"):
            ops.execute_insert(
                sa.text("INSERT OR IGNORE INTO test_rows (id, value) VALUES ('row1', 'dup')"),
                context="my context",
            )


class TestExecuteUpdate:
    def test_update_succeeds(
        self, ops: DatabaseOps, test_table: sa.Table
    ) -> None:
        """Normal update with rowcount > 0 does not raise."""
        ops.execute_insert(
            test_table.insert().values(id="row1", value="old"),
        )
        ops.execute_update(
            test_table.update().where(test_table.c.id == "row1").values(value="new"),
            context="test update",
        )

    def test_update_nonexistent_row_raises_audit_integrity_error(
        self, ops: DatabaseOps, test_table: sa.Table
    ) -> None:
        """Update targeting zero rows must crash with AuditIntegrityError."""
        with pytest.raises(AuditIntegrityError, match="zero rows affected"):
            ops.execute_update(
                test_table.update()
                .where(test_table.c.id == "nonexistent")
                .values(value="new"),
            )

    def test_update_context_in_error_message(
        self, ops: DatabaseOps, test_table: sa.Table
    ) -> None:
        """Context string appears in the AuditIntegrityError message."""
        with pytest.raises(AuditIntegrityError, match="my context"):
            ops.execute_update(
                test_table.update()
                .where(test_table.c.id == "nonexistent")
                .values(value="new"),
                context="my context",
            )


class TestExecuteFetchone:
    def test_returns_row(
        self, ops: DatabaseOps, test_table: sa.Table
    ) -> None:
        ops.execute_insert(
            test_table.insert().values(id="row1", value="hello"),
        )
        row = ops.execute_fetchone(
            sa.select(test_table).where(test_table.c.id == "row1")
        )
        assert row is not None
        assert row.id == "row1"
        assert row.value == "hello"

    def test_returns_none_when_not_found(
        self, ops: DatabaseOps, test_table: sa.Table
    ) -> None:
        row = ops.execute_fetchone(
            sa.select(test_table).where(test_table.c.id == "nonexistent")
        )
        assert row is None


class TestExecuteFetchall:
    def test_returns_all_rows(
        self, ops: DatabaseOps, test_table: sa.Table
    ) -> None:
        ops.execute_insert(test_table.insert().values(id="a", value="1"))
        ops.execute_insert(test_table.insert().values(id="b", value="2"))
        rows = ops.execute_fetchall(sa.select(test_table))
        assert len(rows) == 2

    def test_returns_empty_list_when_no_rows(
        self, ops: DatabaseOps, test_table: sa.Table
    ) -> None:
        rows = ops.execute_fetchall(sa.select(test_table))
        assert rows == []
```

**Implementer note:** `INSERT OR IGNORE` with a duplicate PK on SQLite yields `rowcount=0` without raising a SQL error — this lets us test the actual `execute_insert()` guard. The tests must call `ops.execute_insert()` (not reimplement the guard logic).

- [ ] **Step 2: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/unit/core/landscape/test_database_ops.py -v`
Expected: ALL PASS

- [ ] **Step 3: Commit**

```bash
git add tests/unit/core/landscape/test_database_ops.py
git commit -m "test: add dedicated tests for _database_ops.py — Tier 1 write guard coverage"
```

---

## Task 3: Verify and close Filigree tracking issues

**Rationale:** These fixes address existing tracked issues. Close them when complete.

**Files:** None (Filigree operations only)

- [ ] **Step 1: Close or update the `allow_nan` gap**

The `allow_nan=False` gap wasn't tracked yet. Create a filigree issue and immediately close it:

```bash
# Create and close — documents the fix in the project history
filigree create "Missing allow_nan=False in 6 data_flow_repository json.dumps calls" --type=bug --priority=1
filigree close <id> --reason="Fixed: added allow_nan=False to all 6 audit-path json.dumps calls"
```

- [ ] **Step 2: Close `_database_ops` test coverage issue**

```bash
filigree close elspeth-ce06ab1a68 --reason="Fixed: added dedicated unit tests for all 4 DatabaseOps methods including rowcount=0 crash guard"
```

- [ ] **Step 3: Run full test suite**

Run: `.venv/bin/python -m pytest tests/ -x -q`
Expected: ALL PASS

---

## Scope Notes

### What this plan covers
- `allow_nan=False` on all 6 unguarded `json.dumps` calls in `data_flow_repository.py`
- AST-based audit test proving no unguarded calls remain
- Dedicated unit tests for all 4 `DatabaseOps` methods
- Rowcount==0 crash guard tests for `execute_insert` and `execute_update`
- Filigree issue tracking

### What this plan does NOT cover (future work)
- **Exporter integration tests against real database** (`elspeth-856f4e5e56`) — the unit tests (1139 lines, 50+ tests) are comprehensive; the integration gap is about batch query correctness with realistic data volumes. Separate effort.
- **Checkpoint round-trip tests** (`elspeth-3b3b0c0d8f`) — `test_serialization.py` has 150+ lines of round-trip tests. The gap is about `from_dict` validation paths. Separate effort.
- **Lineage integrity checks** (`elspeth-13c59a43c6`) — separate concern, different subsystem.
