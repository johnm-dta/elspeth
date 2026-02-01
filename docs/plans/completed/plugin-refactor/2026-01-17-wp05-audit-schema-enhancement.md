# WP-05: Audit Schema Enhancement

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add missing columns and fix types for audit completeness. This enables proper aggregation trigger tracking and retry deduplication.

**Architecture:** The Landscape audit system needs two enhancements:
1. `TriggerType` enum to record WHY an aggregation batch was triggered
2. `idempotency_key` column on artifacts for retry deduplication
3. Fix `Batch.status` type from `str` to `BatchStatus` enum

**Tech Stack:** Python 3.12, SQLAlchemy Core, Alembic, Pydantic

**Unlocks:** WP-06 (Aggregation Triggers)

---

## Pre-Flight Checks

Before starting implementation, verify the following prerequisites are in place.

### Check 1: Alembic Configuration

Verify Alembic is properly configured for migration generation:

```bash
# Check alembic.ini exists and has correct database URL
ls -la alembic.ini

# Check env.py imports our metadata
head -50 alembic/env.py | grep -E "(import|metadata)"

# Verify current migration state
alembic current
```

**Expected:**
- `alembic.ini` exists with valid `sqlalchemy.url`
- `alembic/env.py` imports `metadata` from `elspeth.core.landscape.schema`
- `alembic current` shows current revision (or empty for new DB)

**If Alembic is not configured:**
```bash
# Initialize Alembic (only if not already done)
alembic init alembic

# Then edit alembic/env.py to import our metadata:
# from elspeth.core.landscape.schema import metadata
# target_metadata = metadata
```

### Check 2: Test Files Exist

Verify test files exist, create if missing:

```bash
# Check for test_enums.py
ls tests/contracts/test_enums.py 2>/dev/null || echo "MISSING: tests/contracts/test_enums.py"

# Check for test_schema.py
ls tests/core/landscape/test_schema.py 2>/dev/null || echo "MISSING: tests/core/landscape/test_schema.py"

# Ensure directories exist
mkdir -p tests/contracts tests/core/landscape
```

**If test files are missing, create them:**

`tests/contracts/test_enums.py` (if missing):
```python
"""Tests for contract enums."""

from dataclasses import fields


# Existing enum tests go here, WP-05 tests added below
```

`tests/core/landscape/test_schema.py` (if missing):
```python
"""Tests for Landscape schema definitions."""

from dataclasses import fields


# WP-05 schema tests added below
```

### Check 3: Verify Pre-requisite Enums Exist

```bash
# BatchStatus enum must exist (used in Task 4)
python -c "from elspeth.contracts.enums import BatchStatus; print([s.value for s in BatchStatus])"
```

**Expected output:** `['draft', 'executing', 'completed', 'failed']`

---

## Architecture Note: Why models.py, Not audit.py

The Landscape audit models live in two locations:

| File | Purpose |
|------|---------|
| `src/elspeth/contracts/audit.py` | **Read-only query results** - dataclasses returned by explain() queries |
| `src/elspeth/core/landscape/models.py` | **Database models** - dataclasses for CRUD operations |

WP-05 modifies **models.py** because we're changing what gets stored in the database.
The `Artifact` and `Batch` models in models.py are the authoritative definitions.

---

## Task 1: Add TriggerType enum

**Files:**
- Modify: `src/elspeth/contracts/enums.py`
- Test: `tests/contracts/test_enums.py`

**Step 1: Write the failing test**

Add to `tests/contracts/test_enums.py`:

```python
class TestTriggerType:
    """Tests for TriggerType enum."""

    def test_trigger_type_exists(self) -> None:
        """TriggerType can be imported."""
        from elspeth.contracts.enums import TriggerType

        assert TriggerType is not None

    def test_trigger_type_values(self) -> None:
        """TriggerType has all required values."""
        from elspeth.contracts.enums import TriggerType

        assert TriggerType.COUNT.value == "count"
        assert TriggerType.TIMEOUT.value == "timeout"
        assert TriggerType.CONDITION.value == "condition"
        assert TriggerType.END_OF_SOURCE.value == "end_of_source"
        assert TriggerType.MANUAL.value == "manual"

    def test_trigger_type_is_str_enum(self) -> None:
        """TriggerType can be used as string (for database serialization)."""
        from elspeth.contracts.enums import TriggerType

        # str(Enum) should return the value for (str, Enum) types
        assert TriggerType.COUNT == "count"
        assert f"trigger: {TriggerType.TIMEOUT}" == "trigger: timeout"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/contracts/test_enums.py::TestTriggerType::test_trigger_type_exists -v`

Expected: FAIL with `ImportError: cannot import name 'TriggerType'`

**Step 3: Implement TriggerType**

Add to `src/elspeth/contracts/enums.py` after `BatchStatus`:

```python
class TriggerType(str, Enum):
    """Type of trigger that caused an aggregation batch to execute.

    Uses (str, Enum) for database serialization to batches.trigger_type.

    Values:
        COUNT: Batch reached configured row count threshold
        TIMEOUT: Batch reached configured time limit
        CONDITION: Custom condition expression evaluated to true
        END_OF_SOURCE: Source exhausted, flush remaining rows
        MANUAL: Explicitly triggered via API/CLI
    """

    COUNT = "count"
    TIMEOUT = "timeout"
    CONDITION = "condition"
    END_OF_SOURCE = "end_of_source"
    MANUAL = "manual"
```

**Step 4: Export from contracts/__init__.py**

Add `TriggerType` to the exports in `src/elspeth/contracts/__init__.py`:

```python
from elspeth.contracts.enums import (
    ...,
    TriggerType,  # Add this
)

__all__ = [
    ...,
    "TriggerType",  # Add this
]
```

**Step 5: Run tests to verify they pass**

Run: `pytest tests/contracts/test_enums.py::TestTriggerType -v`

Expected: All 3 tests pass

**Step 6: Commit**

```
git add -A && git commit -m "feat(contracts): add TriggerType enum for aggregation triggers

Defines the types of events that can trigger an aggregation batch:
COUNT, TIMEOUT, CONDITION, END_OF_SOURCE, MANUAL.

Used by batches.trigger_type column (WP-05) and config validation (WP-06).

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 2: Add idempotency_key to artifacts table

**Files:**
- Modify: `src/elspeth/core/landscape/schema.py`
- Modify: `src/elspeth/core/landscape/models.py`
- Test: `tests/core/landscape/test_schema.py`

**Step 1: Write the failing test**

Add to schema tests:

```python
def test_artifacts_table_has_idempotency_key() -> None:
    """artifacts table should have idempotency_key column."""
    from elspeth.core.landscape.schema import artifacts_table

    column_names = [c.name for c in artifacts_table.columns]
    assert "idempotency_key" in column_names

def test_artifact_model_has_idempotency_key() -> None:
    """Artifact model should have idempotency_key field."""
    from elspeth.core.landscape.models import Artifact

    # Dataclass fields
    field_names = [f.name for f in fields(Artifact)]
    assert "idempotency_key" in field_names
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/core/landscape/test_schema.py -v -k "idempotency"`

Expected: FAIL

**Step 3: Add column to schema.py**

In `src/elspeth/core/landscape/schema.py`, add to `artifacts_table` (around line 176):

```python
artifacts_table = Table(
    "artifacts",
    metadata,
    Column("artifact_id", String(64), primary_key=True),
    Column("run_id", String(64), ForeignKey("runs.run_id"), nullable=False),
    Column(
        "produced_by_state_id",
        String(64),
        ForeignKey("node_states.state_id"),
        nullable=False,
    ),
    Column("artifact_type", String(32), nullable=False),
    Column("path_or_uri", String(512)),
    Column("content_hash", String(64)),
    Column("size_bytes", BigInteger),
    Column("idempotency_key", String(256)),  # ADD THIS - for retry deduplication
    Column("created_at", DateTime, nullable=False),
    Column("metadata_json", Text),  # JSON blob for type-specific metadata
)
```

**Step 4: Add field to Artifact model**

In `src/elspeth/core/landscape/models.py`, find the `Artifact` dataclass and add:

```python
@dataclass
class Artifact:
    """An output artifact produced by a sink."""

    artifact_id: str
    run_id: str
    produced_by_state_id: str
    artifact_type: str
    created_at: datetime
    path_or_uri: str | None = None
    content_hash: str | None = None
    size_bytes: int | None = None
    idempotency_key: str | None = None  # ADD THIS
    metadata_json: str | None = None
```

**Step 5: Run tests to verify they pass**

Run: `pytest tests/core/landscape/test_schema.py -v -k "idempotency"`

**Step 6: Commit**

```
git add -A && git commit -m "feat(landscape): add idempotency_key to artifacts table

Enables retry deduplication - sinks can set a key to prevent duplicate
artifacts when a row is retried. Schema and model updated.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 3: Add trigger_type to batches table

**Files:**
- Modify: `src/elspeth/core/landscape/schema.py`
- Modify: `src/elspeth/core/landscape/models.py`

**Step 1: Write the failing test**

Add to schema tests:

```python
def test_batches_table_has_trigger_type() -> None:
    """batches table should have trigger_type column."""
    from elspeth.core.landscape.schema import batches_table

    column_names = [c.name for c in batches_table.columns]
    assert "trigger_type" in column_names

def test_batch_model_has_trigger_type() -> None:
    """Batch model should have trigger_type field."""
    from elspeth.contracts.enums import TriggerType
    from elspeth.core.landscape.models import Batch

    field_names = [f.name for f in fields(Batch)]
    assert "trigger_type" in field_names
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/core/landscape/test_schema.py -v -k "trigger_type"`

**Step 3: Add column to batches_table in schema.py**

Find `batches_table` and add the column:

```python
Column("trigger_type", String(32)),  # ADD THIS - TriggerType enum value
```

**Step 4: Add field to Batch model**

In models.py, update the `Batch` dataclass:

```python
@dataclass
class Batch:
    """An aggregation batch collecting tokens."""

    batch_id: str
    run_id: str
    aggregation_node_id: str
    attempt: int
    status: str  # Will be fixed to BatchStatus in Task 4
    created_at: datetime
    aggregation_state_id: str | None = None
    trigger_reason: str | None = None
    trigger_type: str | None = None  # ADD THIS - TriggerType enum value
    completed_at: datetime | None = None
```

**Step 5: Run tests**

Run: `pytest tests/core/landscape/test_schema.py -v -k "trigger_type"`

**Step 6: Commit**

```
git add -A && git commit -m "feat(landscape): add trigger_type to batches table

Records WHY an aggregation batch was triggered (count, timeout, condition,
end_of_source, manual). Used by WP-06 for config-driven aggregation.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 4: Fix Batch.status type from str to BatchStatus

**Files:**
- Modify: `src/elspeth/core/landscape/models.py`
- Test: Verify mypy passes

**Step 1: Write the failing test**

```python
def test_batch_status_is_typed() -> None:
    """Batch.status should accept BatchStatus enum."""
    from elspeth.contracts.enums import BatchStatus
    from elspeth.core.landscape.models import Batch
    from datetime import datetime, timezone

    batch = Batch(
        batch_id="b1",
        run_id="r1",
        aggregation_node_id="agg1",
        attempt=1,
        status=BatchStatus.DRAFT,  # Should work without type error
        created_at=datetime.now(timezone.utc),
    )

    assert batch.status == BatchStatus.DRAFT
```

**Step 2: Update Batch model**

In `src/elspeth/core/landscape/models.py`:

1. Add import at top:
```python
from elspeth.contracts.enums import BatchStatus
```

2. Change the `status` field type:
```python
@dataclass
class Batch:
    """An aggregation batch collecting tokens."""

    batch_id: str
    run_id: str
    aggregation_node_id: str
    attempt: int
    status: BatchStatus  # CHANGED from str
    created_at: datetime
    aggregation_state_id: str | None = None
    trigger_reason: str | None = None
    trigger_type: str | None = None
    completed_at: datetime | None = None
```

**Step 3: Fix any call sites**

Search for places creating `Batch` objects and update them to use `BatchStatus`:

```bash
grep -r "Batch(" src/elspeth/core/landscape/ --include="*.py"
```

Update each to use `status=BatchStatus.DRAFT` (or appropriate status).

**Step 4: Run mypy**

```bash
mypy src/elspeth/core/landscape/models.py --strict
```

**Step 5: Run tests**

```bash
pytest tests/core/landscape/ -v
```

**Step 6: Commit**

```
git add -A && git commit -m "fix(models): change Batch.status from str to BatchStatus

Type safety improvement - Batch.status now requires BatchStatus enum
instead of arbitrary strings. Matches the BatchStatus enum defined
in contracts/enums.py.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 5: Generate Alembic migration

**Files:**
- Create: `alembic/versions/XXXX_wp05_audit_schema_enhancement.py`

**Step 1: Generate migration**

```bash
cd /home/john/elspeth-rapid
alembic revision --autogenerate -m "WP-05: Add idempotency_key to artifacts, trigger_type to batches"
```

**Step 2: Review generated migration**

The migration should include:
- `op.add_column('artifacts', sa.Column('idempotency_key', sa.String(256), nullable=True))`
- `op.add_column('batches', sa.Column('trigger_type', sa.String(32), nullable=True))`

**Troubleshooting: If autogenerate produces empty migration**

If the migration `upgrade()` function is empty, Alembic isn't detecting changes. Common causes:

1. **Metadata not imported:** Check `alembic/env.py` has:
   ```python
   from elspeth.core.landscape.schema import metadata
   target_metadata = metadata
   ```

2. **Table already exists with columns:** If running against existing DB, columns may already exist.
   Check with: `alembic upgrade head --sql` to see what SQL would run.

3. **Wrong database URL:** Check `alembic.ini` points to correct dev database.

**If autogenerate fails, create manual migration:**
```bash
alembic revision -m "WP-05: Add idempotency_key to artifacts, trigger_type to batches"
```

Then edit the generated file:
```python
def upgrade() -> None:
    op.add_column('artifacts', sa.Column('idempotency_key', sa.String(256), nullable=True))
    op.add_column('batches', sa.Column('trigger_type', sa.String(32), nullable=True))

def downgrade() -> None:
    op.drop_column('batches', 'trigger_type')
    op.drop_column('artifacts', 'idempotency_key')
```

**Step 3: Test migration**

```bash
# Apply migration
alembic upgrade head

# Verify columns exist (SQLite example)
sqlite3 landscape.db ".schema artifacts" | grep idempotency_key
sqlite3 landscape.db ".schema batches" | grep trigger_type

# Or for PostgreSQL:
# psql -d elspeth -c "\d artifacts" | grep idempotency_key
# psql -d elspeth -c "\d batches" | grep trigger_type
```

**Step 4: Commit**

```
git add -A && git commit -m "chore(alembic): migration for WP-05 schema changes

Adds:
- artifacts.idempotency_key (String(256)) for retry deduplication
- batches.trigger_type (String(32)) for aggregation trigger tracking

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 6: Run full verification

**Step 1: Run mypy**

```bash
mypy src/elspeth/contracts/enums.py src/elspeth/core/landscape/schema.py src/elspeth/core/landscape/models.py --strict
```

**Step 2: Run all affected tests**

```bash
pytest tests/contracts/test_enums.py tests/core/landscape/ -v
```

**Step 3: Verify exports**

```python
# In Python REPL or test
from elspeth.contracts import TriggerType, BatchStatus
from elspeth.core.landscape.models import Artifact, Batch

# Should work without errors
assert TriggerType.COUNT == "count"
assert BatchStatus.DRAFT == "draft"
```

**Step 4: Final commit**

```
git add -A && git commit -m "chore: verify WP-05 audit schema enhancement complete

- TriggerType enum added to contracts
- idempotency_key added to artifacts table
- trigger_type added to batches table
- Batch.status typed as BatchStatus
- Alembic migration generated and tested
- All tests pass

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Verification Checklist

### Pre-Flight (before starting)

- [ ] Alembic configured (`alembic.ini` exists, `env.py` imports metadata)
- [ ] Test directories exist (`tests/contracts/`, `tests/core/landscape/`)
- [ ] `BatchStatus` enum importable from `elspeth.contracts.enums`

### Implementation (after each task)

- [ ] `TriggerType` enum exists with 5 values (COUNT, TIMEOUT, CONDITION, END_OF_SOURCE, MANUAL)
- [ ] `TriggerType` exported from `elspeth.contracts`
- [ ] `artifacts_table` has `idempotency_key` column (String(256))
- [ ] `Artifact` model has `idempotency_key` field
- [ ] `batches_table` has `trigger_type` column (String(32))
- [ ] `Batch` model has `trigger_type` field
- [ ] `Batch.status` type is `BatchStatus` (not `str`)
- [ ] Alembic migration generated and tested
- [ ] `mypy --strict` passes on contracts and landscape modules
- [ ] All landscape tests pass

---

## Files Changed Summary

| File | Change Type | Description |
|------|-------------|-------------|
| `src/elspeth/contracts/enums.py` | MODIFY | Add TriggerType enum |
| `src/elspeth/contracts/__init__.py` | MODIFY | Export TriggerType |
| `src/elspeth/core/landscape/schema.py` | MODIFY | Add idempotency_key and trigger_type columns |
| `src/elspeth/core/landscape/models.py` | MODIFY | Add fields, fix Batch.status type |
| `tests/contracts/test_enums.py` | MODIFY | Add TriggerType tests |
| `tests/core/landscape/test_schema.py` | MODIFY | Add column tests |
| `alembic/versions/XXXX_*.py` | CREATE | Migration for schema changes |

---

## Dependency Notes

- **Depends on:** Nothing
- **Unlocks:** WP-06 (Aggregation Triggers) - uses TriggerType for config validation
- **Risk:** Low - additive schema changes, nullable columns
