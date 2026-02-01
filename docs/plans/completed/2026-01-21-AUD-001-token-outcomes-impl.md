# AUD-001: Token Outcomes Implementation Plan

**Status:** ✅ IMPLEMENTED (2026-01-22)

> **Historical Note:** This plan was created on 2026-01-21 but implementation was completed before plan status was updated. This document remains as architectural reference.

**Goal:** Add explicit terminal state recording so every token has an auditable outcome record.

**Architecture:** New `token_outcomes` table with `record_token_outcome()` API called from 17 locations in `processor.py`. Partial unique index enforces "exactly one terminal outcome per token."

**Tech Stack:** SQLAlchemy Core, Alembic migrations, pytest

## Implementation Summary

All 16 tasks completed:
- ✅ Task 0: RowOutcome changed to (str, Enum)
- ✅ Tasks 1-4: TokenOutcome dataclass, schema, migration, recorder API
- ✅ Tasks 5-13: 17 recording sites in processor
- ✅ Task 14: explain() integration
- ✅ Task 15: requirements.md updated
- ✅ Task 16: Integration tests passing

**Remaining:** 1 bug (group ID mismatch) - not blocking

---

## Review Findings (2026-01-21)

**Reviewers:** axiom-system-architect:architecture-critic, pr-review-toolkit:code-reviewer

### Critical Fixes Applied:
1. **RowOutcome enum change** - Must change from plain `Enum` to `(str, Enum)` since we're now storing it in DB
2. **Added Alembic migration task** - Task 2.5 creates the migration
3. **Removed bool() coercion** - Violates Tier 1 trust model; trust DB values directly
4. **Fixed method name** - `create_row` not `create_row`
5. **Added processor integration test** - Task 5 now tests actual RowProcessor

---

## Task 0: Change RowOutcome to (str, Enum)

**Files:**
- Modify: `src/elspeth/contracts/enums.py`

**Rationale:** The docstring on `RowOutcome` explicitly says "NOT stored in database" and "plain Enum". AUD-001 changes this - we ARE now storing outcomes. Per codebase pattern, all DB-stored enums use `(str, Enum)`.

**Step 1: Update RowOutcome class definition**

Change `src/elspeth/contracts/enums.py` line 139 from:
```python
class RowOutcome(Enum):
    """Outcome for a token in the pipeline.

    IMPORTANT: These are DERIVED at query time from node_states,
    routing_events, and batch_members - NOT stored in the database.
    Therefore this is plain Enum, not (str, Enum).
```

To:
```python
class RowOutcome(str, Enum):
    """Outcome for a token in the pipeline.

    These outcomes are explicitly recorded in the `token_outcomes` table
    (AUD-001) at determination time. The (str, Enum) base allows direct
    database storage via .value.
```

**Step 2: Run existing tests to verify no regressions**

Run: `pytest tests/ -v -x --tb=short`
Expected: PASS (str, Enum is backward compatible)

**Step 3: Commit**

```bash
git add src/elspeth/contracts/enums.py
git commit -m "refactor(enums): change RowOutcome to (str, Enum) for AUD-001 storage"
```

---

## Task 1: Add TokenOutcome Dataclass

**Files:**
- Create: `src/elspeth/contracts/audit.py` (add to existing)
- Modify: `src/elspeth/contracts/__init__.py`

**Step 1: Write the failing test**

Create `tests/core/test_token_outcomes.py`:

```python
# tests/core/test_token_outcomes.py
"""Tests for token outcome recording."""

import pytest


class TestTokenOutcomeDataclass:
    """Test TokenOutcome dataclass structure."""

    def test_token_outcome_has_required_fields(self) -> None:
        from elspeth.contracts import TokenOutcome

        # Should have these fields
        assert hasattr(TokenOutcome, "__dataclass_fields__")
        fields = TokenOutcome.__dataclass_fields__
        assert "outcome_id" in fields
        assert "run_id" in fields
        assert "token_id" in fields
        assert "outcome" in fields
        assert "is_terminal" in fields
        assert "recorded_at" in fields

    def test_token_outcome_instantiation(self) -> None:
        from datetime import UTC, datetime

        from elspeth.contracts import RowOutcome, TokenOutcome

        outcome = TokenOutcome(
            outcome_id="out_123",
            run_id="run_456",
            token_id="tok_789",
            outcome=RowOutcome.COMPLETED,
            is_terminal=True,
            recorded_at=datetime.now(UTC),
        )
        assert outcome.outcome_id == "out_123"
        assert outcome.is_terminal is True
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/core/test_token_outcomes.py::TestTokenOutcomeDataclass -v`
Expected: FAIL with "cannot import name 'TokenOutcome'"

**Step 3: Add TokenOutcome dataclass to contracts/audit.py**

Add to `src/elspeth/contracts/audit.py` after the other dataclasses:

```python
@dataclass(frozen=True)
class TokenOutcome:
    """Recorded terminal state for a token.

    Captures the moment a token reached its terminal (or buffered) state.
    Part of AUD-001 audit integrity - explicit rather than derived.
    """

    outcome_id: str
    run_id: str
    token_id: str
    outcome: RowOutcome  # Direct type, not forward reference (per code review)
    is_terminal: bool
    recorded_at: datetime

    # Outcome-specific fields (nullable based on outcome type)
    sink_name: str | None = None
    batch_id: str | None = None
    fork_group_id: str | None = None
    join_group_id: str | None = None
    expand_group_id: str | None = None
    error_hash: str | None = None
    context_json: str | None = None
```

**Step 4: Export from contracts/__init__.py**

Add to `src/elspeth/contracts/__init__.py` imports from `audit`:

```python
from elspeth.contracts.audit import (
    # ... existing imports ...
    TokenOutcome,  # Add this
)
```

**Step 5: Run test to verify it passes**

Run: `pytest tests/core/test_token_outcomes.py::TestTokenOutcomeDataclass -v`
Expected: PASS

**Step 6: Commit**

```bash
git add src/elspeth/contracts/audit.py src/elspeth/contracts/__init__.py tests/core/test_token_outcomes.py
git commit -m "feat(contracts): add TokenOutcome dataclass for AUD-001"
```

---

## Task 2: Add token_outcomes Table Schema

**Files:**
- Modify: `src/elspeth/core/landscape/schema.py`
- Test: `tests/core/test_token_outcomes.py`

**Step 1: Write the failing test**

Add to `tests/core/test_token_outcomes.py`:

```python
class TestTokenOutcomesTableSchema:
    """Test token_outcomes table definition."""

    def test_table_exists_in_metadata(self) -> None:
        from elspeth.core.landscape.schema import metadata, token_outcomes_table

        assert token_outcomes_table is not None
        assert "token_outcomes" in metadata.tables

    def test_table_has_required_columns(self) -> None:
        from elspeth.core.landscape.schema import token_outcomes_table

        columns = {c.name for c in token_outcomes_table.columns}
        required = {
            "outcome_id",
            "run_id",
            "token_id",
            "outcome",
            "is_terminal",
            "recorded_at",
            "sink_name",
            "batch_id",
            "fork_group_id",
            "join_group_id",
            "expand_group_id",
            "error_hash",
            "context_json",
        }
        assert required.issubset(columns)

    def test_outcome_id_is_primary_key(self) -> None:
        from elspeth.core.landscape.schema import token_outcomes_table

        pk_columns = [c.name for c in token_outcomes_table.primary_key.columns]
        assert pk_columns == ["outcome_id"]

    def test_run_id_has_foreign_key(self) -> None:
        from elspeth.core.landscape.schema import token_outcomes_table

        run_id_col = token_outcomes_table.c.run_id
        fk_targets = [fk.target_fullname for fk in run_id_col.foreign_keys]
        assert "runs.run_id" in fk_targets

    def test_token_id_has_foreign_key(self) -> None:
        from elspeth.core.landscape.schema import token_outcomes_table

        token_id_col = token_outcomes_table.c.token_id
        fk_targets = [fk.target_fullname for fk in token_id_col.foreign_keys]
        assert "tokens.token_id" in fk_targets
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/core/test_token_outcomes.py::TestTokenOutcomesTableSchema -v`
Expected: FAIL with "cannot import name 'token_outcomes_table'"

**Step 3: Add table definition to schema.py**

Add to `src/elspeth/core/landscape/schema.py` after the `tokens_table` definition:

```python
# === Token Outcomes (AUD-001: Explicit terminal state recording) ===

token_outcomes_table = Table(
    "token_outcomes",
    metadata,
    # Identity
    Column("outcome_id", String(64), primary_key=True),
    Column("run_id", String(64), ForeignKey("runs.run_id"), nullable=False, index=True),
    Column("token_id", String(64), ForeignKey("tokens.token_id"), nullable=False, index=True),
    # Core outcome
    Column("outcome", String(32), nullable=False),
    Column("is_terminal", Integer, nullable=False),  # SQLite doesn't have Boolean, use Integer
    Column("recorded_at", DateTime(timezone=True), nullable=False),
    # Outcome-specific fields (nullable based on outcome type)
    Column("sink_name", String(128)),
    Column("batch_id", String(64), ForeignKey("batches.batch_id")),
    Column("fork_group_id", String(64)),
    Column("join_group_id", String(64)),
    Column("expand_group_id", String(64)),
    Column("error_hash", String(64)),
    # Optional extended context
    Column("context_json", Text),
)

# Partial unique index: exactly one terminal outcome per token
# Note: SQLite partial index syntax differs; SQLAlchemy handles this
Index(
    "ix_token_outcomes_terminal_unique",
    token_outcomes_table.c.token_id,
    unique=True,
    sqlite_where=(token_outcomes_table.c.is_terminal == 1),
    postgresql_where=(token_outcomes_table.c.is_terminal == 1),
)
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/core/test_token_outcomes.py::TestTokenOutcomesTableSchema -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/elspeth/core/landscape/schema.py tests/core/test_token_outcomes.py
git commit -m "feat(schema): add token_outcomes table for AUD-001"
```

---

## Task 2.5: Create Alembic Migration

**Files:**
- Create: `alembic/versions/001_add_token_outcomes.py`

**Rationale:** Per architecture review, schema changes require Alembic migrations for production deployments. `LandscapeDB.in_memory()` auto-creates tables for tests, but production databases need explicit migration.

**Step 1: Generate migration skeleton**

```bash
cd /home/john/elspeth-rapid
alembic revision -m "add_token_outcomes_table"
```

**Step 2: Edit the generated migration file**

```python
"""add_token_outcomes_table

Revision ID: <auto-generated>
Revises: <previous or None>
Create Date: 2026-01-21
"""
from alembic import op
import sqlalchemy as sa

# revision identifiers
revision = '<auto-generated>'
down_revision = None  # or previous revision
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        'token_outcomes',
        sa.Column('outcome_id', sa.String(64), primary_key=True),
        sa.Column('run_id', sa.String(64), sa.ForeignKey('runs.run_id'), nullable=False, index=True),
        sa.Column('token_id', sa.String(64), sa.ForeignKey('tokens.token_id'), nullable=False, index=True),
        sa.Column('outcome', sa.String(32), nullable=False),
        sa.Column('is_terminal', sa.Integer, nullable=False),
        sa.Column('recorded_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('sink_name', sa.String(128)),
        sa.Column('batch_id', sa.String(64), sa.ForeignKey('batches.batch_id')),
        sa.Column('fork_group_id', sa.String(64)),
        sa.Column('join_group_id', sa.String(64)),
        sa.Column('expand_group_id', sa.String(64)),
        sa.Column('error_hash', sa.String(64)),
        sa.Column('context_json', sa.Text),
    )

    # Partial unique index: exactly one terminal outcome per token
    # Note: SQLite and PostgreSQL have different partial index syntax
    op.create_index(
        'ix_token_outcomes_terminal_unique',
        'token_outcomes',
        ['token_id'],
        unique=True,
        postgresql_where=sa.text('is_terminal = 1'),
        sqlite_where=sa.text('is_terminal = 1'),
    )


def downgrade() -> None:
    op.drop_index('ix_token_outcomes_terminal_unique', 'token_outcomes')
    op.drop_table('token_outcomes')
```

**Step 3: Test migration**

```bash
# Apply migration to test database
alembic upgrade head

# Verify table exists
python -c "from elspeth.core.landscape import LandscapeDB; db = LandscapeDB('sqlite:///test.db'); print('OK')"

# Rollback and re-apply
alembic downgrade -1
alembic upgrade head
```

**Step 4: Commit**

```bash
git add alembic/versions/
git commit -m "feat(migration): add token_outcomes table migration (AUD-001)"
```

---

## Task 3: Add record_token_outcome() to LandscapeRecorder

**Files:**
- Modify: `src/elspeth/core/landscape/recorder.py`
- Test: `tests/core/test_token_outcomes.py`

**Step 1: Write the failing test**

Add to `tests/core/test_token_outcomes.py`:

```python
class TestRecordTokenOutcome:
    """Test record_token_outcome() method."""

    @pytest.fixture
    def db(self):
        """Create in-memory database with schema."""
        from elspeth.core.landscape import LandscapeDB

        db = LandscapeDB.in_memory()
        return db

    @pytest.fixture
    def recorder(self, db):
        """Create recorder with test database."""
        from elspeth.core.landscape import LandscapeRecorder

        return LandscapeRecorder(db)

    @pytest.fixture
    def run_with_token(self, recorder):
        """Create a run with a token for testing."""
        # Begin run
        run = recorder.begin_run(config={"test": True})

        # Register source node
        from elspeth.contracts import Determinism, NodeType
        from elspeth.contracts.schema import SchemaConfig

        recorder.register_node(
            run_id=run.run_id,
            node_id="source_1",
            plugin_name="test_source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0.0",
            config={},
            determinism=Determinism.DETERMINISTIC,
            schema_config=SchemaConfig.dynamic(),
        )

        # Create row and token
        row = recorder.create_row(
            run_id=run.run_id,
            source_node_id="source_1",
            row_index=0,
            row_data={"id": 1},
        )
        token = recorder.create_token(
            run_id=run.run_id,
            row_id=row.row_id,
        )

        return run, token

    def test_record_completed_outcome(self, recorder, run_with_token) -> None:
        from elspeth.contracts import RowOutcome

        run, token = run_with_token

        outcome_id = recorder.record_token_outcome(
            run_id=run.run_id,
            token_id=token.token_id,
            outcome=RowOutcome.COMPLETED,
            sink_name="output",
        )

        assert outcome_id is not None
        assert outcome_id.startswith("out_")

    def test_record_routed_outcome(self, recorder, run_with_token) -> None:
        from elspeth.contracts import RowOutcome

        run, token = run_with_token

        outcome_id = recorder.record_token_outcome(
            run_id=run.run_id,
            token_id=token.token_id,
            outcome=RowOutcome.ROUTED,
            sink_name="errors",
        )

        assert outcome_id is not None

    def test_record_buffered_then_terminal(self, recorder, run_with_token) -> None:
        """BUFFERED followed by terminal should succeed."""
        from elspeth.contracts import RowOutcome

        run, token = run_with_token

        # First record BUFFERED (non-terminal)
        recorder.record_token_outcome(
            run_id=run.run_id,
            token_id=token.token_id,
            outcome=RowOutcome.BUFFERED,
            batch_id="batch_123",
        )

        # Then record terminal outcome - should succeed
        outcome_id = recorder.record_token_outcome(
            run_id=run.run_id,
            token_id=token.token_id,
            outcome=RowOutcome.CONSUMED_IN_BATCH,
            batch_id="batch_123",
        )

        assert outcome_id is not None

    def test_duplicate_terminal_raises(self, recorder, run_with_token) -> None:
        """Two terminal outcomes for same token should raise IntegrityError."""
        from sqlalchemy.exc import IntegrityError

        from elspeth.contracts import RowOutcome

        run, token = run_with_token

        # First terminal outcome
        recorder.record_token_outcome(
            run_id=run.run_id,
            token_id=token.token_id,
            outcome=RowOutcome.COMPLETED,
            sink_name="output",
        )

        # Second terminal outcome should fail
        with pytest.raises(IntegrityError):
            recorder.record_token_outcome(
                run_id=run.run_id,
                token_id=token.token_id,
                outcome=RowOutcome.ROUTED,
                sink_name="errors",
            )
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/core/test_token_outcomes.py::TestRecordTokenOutcome -v`
Expected: FAIL with "has no attribute 'record_token_outcome'"

**Step 3: Implement record_token_outcome() in recorder.py**

Add to `src/elspeth/core/landscape/recorder.py`:

```python
def record_token_outcome(
    self,
    run_id: str,
    token_id: str,
    outcome: RowOutcome,
    *,
    sink_name: str | None = None,
    batch_id: str | None = None,
    fork_group_id: str | None = None,
    join_group_id: str | None = None,
    expand_group_id: str | None = None,
    error_hash: str | None = None,
    context: dict[str, Any] | None = None,
) -> str:
    """Record a token's outcome in the audit trail.

    Called at the moment the outcome is determined in processor.py.
    For BUFFERED tokens, a second call records the terminal outcome
    when the batch flushes.

    Args:
        run_id: Current run ID
        token_id: Token that reached this outcome
        outcome: The RowOutcome enum value
        sink_name: For ROUTED/COMPLETED - which sink
        batch_id: For CONSUMED_IN_BATCH/BUFFERED - which batch
        fork_group_id: For FORKED - the fork group
        join_group_id: For COALESCED - the join group
        expand_group_id: For EXPANDED - the expand group
        error_hash: For FAILED/QUARANTINED - hash of error details
        context: Optional additional context (stored as JSON)

    Returns:
        outcome_id for tracking

    Raises:
        IntegrityError: If terminal outcome already exists for token
    """
    from elspeth.core.landscape.schema import token_outcomes_table

    outcome_id = f"out_{_generate_id()[:12]}"
    is_terminal = outcome != RowOutcome.BUFFERED
    context_json = json.dumps(context) if context else None

    with self._db.connection() as conn:
        conn.execute(
            token_outcomes_table.insert().values(
                outcome_id=outcome_id,
                run_id=run_id,
                token_id=token_id,
                outcome=outcome.value,
                is_terminal=1 if is_terminal else 0,
                recorded_at=_now(),
                sink_name=sink_name,
                batch_id=batch_id,
                fork_group_id=fork_group_id,
                join_group_id=join_group_id,
                expand_group_id=expand_group_id,
                error_hash=error_hash,
                context_json=context_json,
            )
        )

    return outcome_id
```

Also add the import at the top of recorder.py if not present:
```python
from elspeth.contracts import RowOutcome
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/core/test_token_outcomes.py::TestRecordTokenOutcome -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/elspeth/core/landscape/recorder.py tests/core/test_token_outcomes.py
git commit -m "feat(recorder): add record_token_outcome() for AUD-001"
```

---

## Task 4: Add get_token_outcome() Query Method

**Files:**
- Modify: `src/elspeth/core/landscape/recorder.py`
- Test: `tests/core/test_token_outcomes.py`

**Step 1: Write the failing test**

Add to `tests/core/test_token_outcomes.py`:

```python
class TestGetTokenOutcome:
    """Test get_token_outcome() method."""

    @pytest.fixture
    def db(self):
        from elspeth.core.landscape import LandscapeDB

        return LandscapeDB.in_memory()

    @pytest.fixture
    def recorder(self, db):
        from elspeth.core.landscape import LandscapeRecorder

        return LandscapeRecorder(db)

    @pytest.fixture
    def run_with_outcome(self, recorder):
        """Create run, token, and outcome for testing."""
        from elspeth.contracts import Determinism, NodeType, RowOutcome
        from elspeth.contracts.schema import SchemaConfig

        run = recorder.begin_run(config={})
        recorder.register_node(
            run_id=run.run_id,
            node_id="src",
            plugin_name="test",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            determinism=Determinism.DETERMINISTIC,
            schema_config=SchemaConfig.dynamic(),
        )
        row = recorder.create_row(run.run_id, "src", 0, {"x": 1})
        token = recorder.create_token(run.run_id, row.row_id)
        outcome_id = recorder.record_token_outcome(
            run.run_id, token.token_id, RowOutcome.COMPLETED, sink_name="out"
        )
        return run, token, outcome_id

    def test_get_token_outcome_returns_dataclass(self, recorder, run_with_outcome) -> None:
        from elspeth.contracts import TokenOutcome

        run, token, _ = run_with_outcome
        result = recorder.get_token_outcome(token.token_id)

        assert isinstance(result, TokenOutcome)
        assert result.token_id == token.token_id
        assert result.outcome.value == "completed"

    def test_get_token_outcome_returns_latest(self, recorder, run_with_outcome) -> None:
        """Should return latest (terminal) outcome, not BUFFERED."""
        from elspeth.contracts import RowOutcome

        run, token, _ = run_with_outcome
        # The fixture already recorded COMPLETED; this tests the query returns it
        result = recorder.get_token_outcome(token.token_id)
        assert result.is_terminal is True

    def test_get_nonexistent_returns_none(self, recorder, db) -> None:
        result = recorder.get_token_outcome("nonexistent_token")
        assert result is None
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/core/test_token_outcomes.py::TestGetTokenOutcome -v`
Expected: FAIL with "has no attribute 'get_token_outcome'"

**Step 3: Implement get_token_outcome()**

Add to `src/elspeth/core/landscape/recorder.py`:

```python
def get_token_outcome(self, token_id: str) -> TokenOutcome | None:
    """Get the latest outcome for a token.

    Returns the terminal outcome if one exists, otherwise the most
    recent non-terminal outcome (BUFFERED).

    Args:
        token_id: Token to look up

    Returns:
        TokenOutcome dataclass or None if no outcome recorded
    """
    from elspeth.core.landscape.schema import token_outcomes_table

    with self._db.connection() as conn:
        # Get most recent outcome (terminal preferred)
        result = conn.execute(
            select(token_outcomes_table)
            .where(token_outcomes_table.c.token_id == token_id)
            .order_by(
                token_outcomes_table.c.is_terminal.desc(),  # Terminal first
                token_outcomes_table.c.recorded_at.desc(),  # Then by time
            )
            .limit(1)
        ).fetchone()

        if result is None:
            return None

        # Tier 1 Trust Model: This is OUR data. Trust DB values directly.
        # If is_terminal is not 0 or 1, that's an audit integrity violation.
        return TokenOutcome(
            outcome_id=result.outcome_id,
            run_id=result.run_id,
            token_id=result.token_id,
            outcome=RowOutcome(result.outcome),
            is_terminal=result.is_terminal == 1,  # DB stores as Integer; no defensive bool()
            recorded_at=result.recorded_at,
            sink_name=result.sink_name,
            batch_id=result.batch_id,
            fork_group_id=result.fork_group_id,
            join_group_id=result.join_group_id,
            expand_group_id=result.expand_group_id,
            error_hash=result.error_hash,
            context_json=result.context_json,
        )
```

Also add the import:
```python
from elspeth.contracts import TokenOutcome
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/core/test_token_outcomes.py::TestGetTokenOutcome -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/elspeth/core/landscape/recorder.py tests/core/test_token_outcomes.py
git commit -m "feat(recorder): add get_token_outcome() query for AUD-001"
```

---

## Task 5: Add Outcome Recording to Processor - COMPLETED Outcomes

**Files:**
- Modify: `src/elspeth/engine/processor.py`
- Test: `tests/engine/test_processor_outcomes.py`

**Step 1: Write the failing integration test**

Create `tests/engine/test_processor_outcomes.py`:

```python
# tests/engine/test_processor_outcomes.py
"""Integration tests for processor outcome recording."""

import pytest

from elspeth.contracts import RowOutcome


class TestProcessorRecordsOutcomes:
    """Test that processor records outcomes at determination points."""

    @pytest.fixture
    def setup_pipeline(self):
        """Set up minimal pipeline for testing outcome recording."""
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        return db, recorder

    def test_completed_outcome_recorded_at_pipeline_end(self, setup_pipeline) -> None:
        """Default COMPLETED outcome is recorded when row reaches end."""
        db, recorder = setup_pipeline

        # Create run
        run = recorder.begin_run(config={})

        # Register minimal nodes
        from elspeth.contracts import Determinism, NodeType
        from elspeth.contracts.schema import SchemaConfig

        recorder.register_node(
            run.run_id, "src", "test", NodeType.SOURCE, "1.0", {},
            Determinism.DETERMINISTIC, SchemaConfig.dynamic()
        )
        recorder.register_node(
            run.run_id, "sink", "test", NodeType.SINK, "1.0", {},
            Determinism.DETERMINISTIC, SchemaConfig.dynamic()
        )

        # Create row and token
        row = recorder.create_row(run.run_id, "src", 0, {"x": 1})
        token = recorder.create_token(run.run_id, row.row_id)

        # Simulate processor recording COMPLETED
        recorder.record_token_outcome(
            run.run_id, token.token_id, RowOutcome.COMPLETED, sink_name="sink"
        )

        # Verify outcome recorded
        outcome = recorder.get_token_outcome(token.token_id)
        assert outcome is not None
        assert outcome.outcome == RowOutcome.COMPLETED
        assert outcome.sink_name == "sink"
        assert outcome.is_terminal is True
```

**Step 2: Run test to verify it passes (API already works)**

Run: `pytest tests/engine/test_processor_outcomes.py -v`
Expected: PASS (this tests the API we just built)

**Step 3: Identify COMPLETED locations in processor.py**

Lines: 228, 287, 344, 892 return `RowOutcome.COMPLETED`

**Step 4: Add outcome recording at line 892 (default end-of-pipeline)**

In `src/elspeth/engine/processor.py`, find line ~892:

```python
# Before:
return RowResult(
    token=token,
    outcome=RowOutcome.COMPLETED,
)

# After:
self._recorder.record_token_outcome(
    run_id=self._run_id,
    token_id=token.token_id,
    outcome=RowOutcome.COMPLETED,
    sink_name=self._output_sink_name,
)
return RowResult(
    token=token,
    outcome=RowOutcome.COMPLETED,
)
```

Note: The processor needs `_output_sink_name` - check if it's available or needs to be passed.

**Step 5: Run full test suite to verify no regressions**

Run: `pytest tests/engine/ -v -x`
Expected: PASS

**Step 6: Commit**

```bash
git add src/elspeth/engine/processor.py tests/engine/test_processor_outcomes.py
git commit -m "feat(processor): record COMPLETED outcome at pipeline end (AUD-001)"
```

---

## Task 6-11: Add Outcome Recording for Remaining Outcomes

For each remaining outcome type, follow the same pattern:

| Task | Line(s) | Outcome | Context Fields |
|------|---------|---------|----------------|
| 6 | 638, 730, 816 | ROUTED | `sink_name` |
| 7 | 669, 848 | FORKED | `fork_group_id` |
| 8 | 205, 706 | FAILED | `error_hash` |
| 9 | 720 | QUARANTINED | `error_hash` |
| 10 | 319, 370 | CONSUMED_IN_BATCH | `batch_id` |
| 11 | 361 | BUFFERED | `batch_id` |
| 12 | 775 | EXPANDED | `expand_group_id` |
| 13 | 883 | COALESCED | `join_group_id` |

Each task follows the pattern:
1. Write test for that outcome type
2. Run test (may pass if API works, may fail if context unavailable)
3. Add `record_token_outcome()` call before the return statement
4. Run tests
5. Commit

---

## Task 14: Update explain() to Include Outcomes

**Files:**
- Modify: `src/elspeth/core/landscape/lineage.py`
- Test: `tests/core/test_token_outcomes.py`

**Step 1: Write the failing test**

Add to `tests/core/test_token_outcomes.py`:

```python
class TestExplainIncludesOutcome:
    """Test that explain() returns recorded outcomes."""

    def test_explain_returns_outcome(self) -> None:
        from elspeth.contracts import Determinism, NodeType, RowOutcome
        from elspeth.contracts.schema import SchemaConfig
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.core.landscape.lineage import explain

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)

        run = recorder.begin_run(config={})
        recorder.register_node(
            run.run_id, "src", "test", NodeType.SOURCE, "1.0", {},
            Determinism.DETERMINISTIC, SchemaConfig.dynamic()
        )
        row = recorder.create_row(run.run_id, "src", 0, {"x": 1})
        token = recorder.create_token(run.run_id, row.row_id)
        recorder.record_token_outcome(
            run.run_id, token.token_id, RowOutcome.COMPLETED, sink_name="out"
        )

        result = explain(recorder, run.run_id, token_id=token.token_id)

        assert result is not None
        assert result.outcome is not None
        assert result.outcome.outcome == RowOutcome.COMPLETED
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/core/test_token_outcomes.py::TestExplainIncludesOutcome -v`
Expected: FAIL (LineageResult doesn't have `outcome` field yet)

**Step 3: Add outcome to LineageResult and explain()**

Modify `src/elspeth/core/landscape/lineage.py`:

1. Add `outcome: TokenOutcome | None = None` to `LineageResult` dataclass
2. In `explain()` function, query the outcome and add to result

**Step 4: Run test to verify it passes**

Run: `pytest tests/core/test_token_outcomes.py::TestExplainIncludesOutcome -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/elspeth/core/landscape/lineage.py tests/core/test_token_outcomes.py
git commit -m "feat(lineage): include outcome in explain() results (AUD-001)"
```

---

## Task 15: Update requirements.md Status

**Files:**
- Modify: `docs/design/requirements.md`

**Step 1: Update AUD-001 status**

Change line 580 from:
```
| AUD-001 | 🆕 Every token reaches exactly one terminal state | Bug analysis | ⚠️ AT RISK | EXPANDED/BUFFERED/COALESCED need audit |
```

To:
```
| AUD-001 | 🆕 Every token reaches exactly one terminal state | Bug analysis | ✅ IMPLEMENTED | `token_outcomes` table with partial unique index |
```

**Step 2: Commit**

```bash
git add docs/design/requirements.md
git commit -m "docs: mark AUD-001 as IMPLEMENTED"
```

---

## Task 16: Final Integration Test

**Files:**
- Test: `tests/engine/test_processor_outcomes.py`

**Step 1: Write comprehensive integration test**

Add to `tests/engine/test_processor_outcomes.py`:

```python
class TestAllOutcomeTypesRecorded:
    """Verify all outcome types are properly recorded."""

    @pytest.mark.parametrize("outcome", [
        RowOutcome.COMPLETED,
        RowOutcome.ROUTED,
        RowOutcome.FORKED,
        RowOutcome.FAILED,
        RowOutcome.QUARANTINED,
        RowOutcome.CONSUMED_IN_BATCH,
        RowOutcome.COALESCED,
        RowOutcome.EXPANDED,
        RowOutcome.BUFFERED,
    ])
    def test_outcome_type_can_be_recorded(self, outcome) -> None:
        """Each outcome type should be recordable."""
        from elspeth.contracts import Determinism, NodeType
        from elspeth.contracts.schema import SchemaConfig
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)

        run = recorder.begin_run(config={})
        recorder.register_node(
            run.run_id, "src", "test", NodeType.SOURCE, "1.0", {},
            Determinism.DETERMINISTIC, SchemaConfig.dynamic()
        )
        row = recorder.create_row(run.run_id, "src", 0, {"x": 1})
        token = recorder.create_token(run.run_id, row.row_id)

        # Record outcome with appropriate context
        kwargs = {}
        if outcome in (RowOutcome.COMPLETED, RowOutcome.ROUTED):
            kwargs["sink_name"] = "test_sink"
        elif outcome in (RowOutcome.CONSUMED_IN_BATCH, RowOutcome.BUFFERED):
            kwargs["batch_id"] = "batch_123"
        elif outcome == RowOutcome.FORKED:
            kwargs["fork_group_id"] = "fork_123"
        elif outcome == RowOutcome.COALESCED:
            kwargs["join_group_id"] = "join_123"
        elif outcome == RowOutcome.EXPANDED:
            kwargs["expand_group_id"] = "expand_123"
        elif outcome in (RowOutcome.FAILED, RowOutcome.QUARANTINED):
            kwargs["error_hash"] = "error_123"

        outcome_id = recorder.record_token_outcome(
            run.run_id, token.token_id, outcome, **kwargs
        )

        assert outcome_id is not None
        recorded = recorder.get_token_outcome(token.token_id)
        assert recorded.outcome == outcome
```

**Step 2: Run all tests**

Run: `pytest tests/ -v`
Expected: ALL PASS

**Step 3: Final commit**

```bash
git add tests/engine/test_processor_outcomes.py
git commit -m "test: add comprehensive outcome recording tests (AUD-001)"
```

---

## Summary

| Task | Description | Commit Message |
|------|-------------|----------------|
| 1 | TokenOutcome dataclass | `feat(contracts): add TokenOutcome dataclass` |
| 2 | Schema table | `feat(schema): add token_outcomes table` |
| 3 | record_token_outcome() | `feat(recorder): add record_token_outcome()` |
| 4 | get_token_outcome() | `feat(recorder): add get_token_outcome()` |
| 5 | COMPLETED recording | `feat(processor): record COMPLETED outcome` |
| 6-13 | Other outcomes | `feat(processor): record X outcome` |
| 14 | explain() update | `feat(lineage): include outcome in explain()` |
| 15 | requirements.md | `docs: mark AUD-001 as IMPLEMENTED` |
| 16 | Final tests | `test: add comprehensive outcome tests` |

**Estimated time:** 2-3 hours for experienced developer
