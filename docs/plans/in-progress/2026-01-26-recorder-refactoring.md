# LandscapeRecorder Refactoring Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Refactor the 2,881-line god-class `LandscapeRecorder` into focused repository modules while preserving audit integrity guarantees.

**Architecture:** Extract nine specialized repositories from the monolithic recorder. Repositories already exist in `repositories.py` but are unused - integrate them first, then add missing ones. Keep `LandscapeRecorder` as a facade for backward compatibility.

**Tech Stack:** Python 3.13, SQLAlchemy Core, Pydantic contracts, pluggy

---

## Critical Context

### Data Manifesto Compliance

- **Tier 1 (Audit DB):** Crash on any anomaly - no coercion, no defaults
- **Tier 3 (External):** Validate at boundary, coerce where needed
- Repositories handle Tier 1 only - they crash on bad data

### Existing Infrastructure

- `repositories.py` has 10 repository classes (load-only, unused)
- `recorder.py` has inline conversions duplicating repository logic
- Pattern: `Repository.__init__(session)` + `load(row) -> DomainObject`

### Files Involved

| File | Current Size | Target Size |
|------|-------------|-------------|
| `recorder.py` | 2,881 lines | ~400 lines |
| `repositories.py` | 235 lines | ~600 lines |
| `_helpers.py` | NEW | ~80 lines |
| `_database_ops.py` | NEW | ~60 lines |

---

## Phase 1: Foundation (Low Risk)

### Task 1.1: Create Database Operations Helper

**Files:**
- Create: `src/elspeth/core/landscape/_database_ops.py`
- Test: `tests/core/landscape/test_database_ops.py`

**Step 1: Write the failing test**

```python
# tests/core/landscape/test_database_ops.py
"""Tests for database operation helpers."""

import pytest
from unittest.mock import MagicMock, patch

from elspeth.core.landscape._database_ops import DatabaseOps


class TestDatabaseOps:
    """Test database operation helper methods."""

    def test_execute_fetchone_returns_row(self) -> None:
        """execute_fetchone returns single row from query."""
        mock_db = MagicMock()
        mock_conn = MagicMock()
        mock_result = MagicMock()
        mock_row = MagicMock(id="row1")

        mock_db.connection.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_db.connection.return_value.__exit__ = MagicMock(return_value=False)
        mock_conn.execute.return_value = mock_result
        mock_result.fetchone.return_value = mock_row

        ops = DatabaseOps(mock_db)
        query = MagicMock()

        result = ops.execute_fetchone(query)

        assert result == mock_row
        mock_conn.execute.assert_called_once_with(query)

    def test_execute_fetchone_returns_none_when_no_row(self) -> None:
        """execute_fetchone returns None when no row found."""
        mock_db = MagicMock()
        mock_conn = MagicMock()
        mock_result = MagicMock()

        mock_db.connection.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_db.connection.return_value.__exit__ = MagicMock(return_value=False)
        mock_conn.execute.return_value = mock_result
        mock_result.fetchone.return_value = None

        ops = DatabaseOps(mock_db)
        result = ops.execute_fetchone(MagicMock())

        assert result is None

    def test_execute_fetchall_returns_list(self) -> None:
        """execute_fetchall returns list of rows."""
        mock_db = MagicMock()
        mock_conn = MagicMock()
        mock_result = MagicMock()
        mock_rows = [MagicMock(id="row1"), MagicMock(id="row2")]

        mock_db.connection.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_db.connection.return_value.__exit__ = MagicMock(return_value=False)
        mock_conn.execute.return_value = mock_result
        mock_result.fetchall.return_value = mock_rows

        ops = DatabaseOps(mock_db)
        result = ops.execute_fetchall(MagicMock())

        assert result == mock_rows

    def test_execute_insert_commits(self) -> None:
        """execute_insert executes insert statement."""
        mock_db = MagicMock()
        mock_conn = MagicMock()

        mock_db.connection.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_db.connection.return_value.__exit__ = MagicMock(return_value=False)

        ops = DatabaseOps(mock_db)
        stmt = MagicMock()

        ops.execute_insert(stmt)

        mock_conn.execute.assert_called_once_with(stmt)
```

**Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/core/landscape/test_database_ops.py -v`
Expected: FAIL with "No module named 'elspeth.core.landscape._database_ops'"

**Step 3: Write minimal implementation**

```python
# src/elspeth/core/landscape/_database_ops.py
"""Database operation helpers to reduce boilerplate in recorder.

Consolidates the repeated `with self._db.connection() as conn:` pattern.
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from elspeth.core.landscape.database import LandscapeDB


class DatabaseOps:
    """Helper for common database operations.

    Reduces boilerplate in recorder methods by centralizing
    connection management.
    """

    def __init__(self, db: "LandscapeDB") -> None:
        self._db = db

    def execute_fetchone(self, query: Any) -> Any | None:
        """Execute query and return single row or None."""
        with self._db.connection() as conn:
            result = conn.execute(query)
            return result.fetchone()

    def execute_fetchall(self, query: Any) -> list[Any]:
        """Execute query and return all rows."""
        with self._db.connection() as conn:
            result = conn.execute(query)
            return list(result.fetchall())

    def execute_insert(self, stmt: Any) -> None:
        """Execute insert statement."""
        with self._db.connection() as conn:
            conn.execute(stmt)

    def execute_update(self, stmt: Any) -> None:
        """Execute update statement."""
        with self._db.connection() as conn:
            conn.execute(stmt)
```

**Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/core/landscape/test_database_ops.py -v`
Expected: PASS (4 tests)

**Step 5: Commit**

```bash
git add src/elspeth/core/landscape/_database_ops.py tests/core/landscape/test_database_ops.py
git commit -m "$(cat <<'EOF'
refactor(landscape): add DatabaseOps helper to reduce connection boilerplate

Consolidates the 53 repeated `with self._db.connection()` patterns
into reusable methods. Foundation for recorder refactoring.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 1.2: Create Common Helpers Module

**Files:**
- Create: `src/elspeth/core/landscape/_helpers.py`
- Test: `tests/core/landscape/test_helpers.py`

**Step 1: Write the failing test**

```python
# tests/core/landscape/test_helpers.py
"""Tests for landscape helper functions."""

from datetime import UTC, datetime
from enum import Enum

import pytest

from elspeth.core.landscape._helpers import generate_id, now, coerce_enum


class TestNow:
    """Tests for now() helper."""

    def test_returns_utc_datetime(self) -> None:
        """now() returns UTC datetime."""
        result = now()
        assert isinstance(result, datetime)
        assert result.tzinfo == UTC

    def test_returns_current_time(self) -> None:
        """now() returns approximately current time."""
        before = datetime.now(UTC)
        result = now()
        after = datetime.now(UTC)
        assert before <= result <= after


class TestGenerateId:
    """Tests for generate_id() helper."""

    def test_returns_hex_string(self) -> None:
        """generate_id() returns hex string."""
        result = generate_id()
        assert isinstance(result, str)
        # UUID4 hex is 32 characters
        assert len(result) == 32
        # All characters are hex
        assert all(c in "0123456789abcdef" for c in result)

    def test_returns_unique_ids(self) -> None:
        """generate_id() returns unique IDs each call."""
        ids = [generate_id() for _ in range(100)]
        assert len(set(ids)) == 100


class SampleEnum(Enum):
    """Sample enum for testing."""
    VALUE_A = "value_a"
    VALUE_B = "value_b"


class TestCoerceEnum:
    """Tests for coerce_enum() helper."""

    def test_returns_enum_unchanged(self) -> None:
        """coerce_enum passes through enum values."""
        result = coerce_enum(SampleEnum.VALUE_A, SampleEnum)
        assert result is SampleEnum.VALUE_A

    def test_converts_string_to_enum(self) -> None:
        """coerce_enum converts valid string to enum."""
        result = coerce_enum("value_a", SampleEnum)
        assert result == SampleEnum.VALUE_A

    def test_crashes_on_invalid_string(self) -> None:
        """coerce_enum crashes on invalid string (Tier 1 trust)."""
        with pytest.raises(ValueError):
            coerce_enum("invalid_value", SampleEnum)
```

**Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/core/landscape/test_helpers.py -v`
Expected: FAIL with "No module named 'elspeth.core.landscape._helpers'"

**Step 3: Write minimal implementation**

```python
# src/elspeth/core/landscape/_helpers.py
"""Common helper functions for landscape modules.

These are extracted from recorder.py to be shared across repositories.
"""

import uuid
from datetime import UTC, datetime
from enum import Enum
from typing import TypeVar

E = TypeVar("E", bound=Enum)


def now() -> datetime:
    """Get current UTC timestamp."""
    return datetime.now(UTC)


def generate_id() -> str:
    """Generate a unique ID (UUID4 hex)."""
    return uuid.uuid4().hex


def coerce_enum(value: str | E, enum_type: type[E]) -> E:
    """Coerce a string or enum value to the target enum type.

    Per Data Manifesto: This is for Tier 1 data (our audit DB).
    Invalid values CRASH - no silent coercion.

    Args:
        value: String or enum value to coerce
        enum_type: Target enum type

    Returns:
        Enum value

    Raises:
        ValueError: If string is not a valid enum value
    """
    if isinstance(value, enum_type):
        return value
    return enum_type(value)
```

**Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/core/landscape/test_helpers.py -v`
Expected: PASS (6 tests)

**Step 5: Commit**

```bash
git add src/elspeth/core/landscape/_helpers.py tests/core/landscape/test_helpers.py
git commit -m "$(cat <<'EOF'
refactor(landscape): extract common helpers from recorder

Extracts now(), generate_id(), and coerce_enum() for reuse
across repository modules. Maintains Tier 1 crash-on-invalid
semantics per Data Manifesto.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 1.3: Add Missing NodeState Repository

**Files:**
- Modify: `src/elspeth/core/landscape/repositories.py`
- Test: `tests/core/landscape/test_repositories.py`

**Step 1: Write the failing test**

Add to existing test file or create new:

```python
# tests/core/landscape/test_node_state_repository.py
"""Tests for NodeStateRepository."""

from datetime import UTC, datetime
from unittest.mock import MagicMock

import pytest

from elspeth.contracts.audit import (
    NodeStateCompleted,
    NodeStateFailed,
    NodeStateOpen,
    NodeStatePending,
)
from elspeth.contracts.enums import NodeStateStatus
from elspeth.core.landscape.repositories import NodeStateRepository


class TestNodeStateRepository:
    """Tests for NodeStateRepository.load()."""

    def test_load_open_state(self) -> None:
        """Load returns NodeStateOpen for OPEN status."""
        row = MagicMock(
            state_id="state_1",
            token_id="token_1",
            node_id="node_1",
            status="open",
            input_hash="hash_in",
            started_at=datetime.now(UTC),
            context_before_json='{"key": "value"}',
            # Fields that should be None for OPEN
            output_hash=None,
            duration_ms=None,
            completed_at=None,
            context_after_json=None,
            error_json=None,
        )

        repo = NodeStateRepository(MagicMock())
        result = repo.load(row)

        assert isinstance(result, NodeStateOpen)
        assert result.state_id == "state_1"
        assert result.status == NodeStateStatus.OPEN

    def test_load_pending_state(self) -> None:
        """Load returns NodeStatePending for PENDING status."""
        row = MagicMock(
            state_id="state_1",
            token_id="token_1",
            node_id="node_1",
            status="pending",
            input_hash="hash_in",
            started_at=datetime.now(UTC),
            context_before_json='{"key": "value"}',
            duration_ms=100,
            # Fields that should be None for PENDING
            output_hash=None,
            completed_at=None,
            context_after_json=None,
            error_json=None,
        )

        repo = NodeStateRepository(MagicMock())
        result = repo.load(row)

        assert isinstance(result, NodeStatePending)
        assert result.duration_ms == 100

    def test_load_completed_state(self) -> None:
        """Load returns NodeStateCompleted for COMPLETED status."""
        completed_at = datetime.now(UTC)
        row = MagicMock(
            state_id="state_1",
            token_id="token_1",
            node_id="node_1",
            status="completed",
            input_hash="hash_in",
            started_at=datetime.now(UTC),
            context_before_json='{"key": "value"}',
            output_hash="hash_out",
            duration_ms=100,
            completed_at=completed_at,
            context_after_json='{"result": "ok"}',
            error_json=None,
        )

        repo = NodeStateRepository(MagicMock())
        result = repo.load(row)

        assert isinstance(result, NodeStateCompleted)
        assert result.output_hash == "hash_out"
        assert result.completed_at == completed_at

    def test_load_failed_state(self) -> None:
        """Load returns NodeStateFailed for FAILED status."""
        row = MagicMock(
            state_id="state_1",
            token_id="token_1",
            node_id="node_1",
            status="failed",
            input_hash="hash_in",
            started_at=datetime.now(UTC),
            context_before_json='{"key": "value"}',
            duration_ms=100,
            completed_at=datetime.now(UTC),
            error_json='{"error": "boom"}',
            # May or may not have output
            output_hash=None,
            context_after_json=None,
        )

        repo = NodeStateRepository(MagicMock())
        result = repo.load(row)

        assert isinstance(result, NodeStateFailed)
        assert result.error_json == '{"error": "boom"}'

    def test_load_crashes_on_pending_without_duration(self) -> None:
        """Load crashes if PENDING state has NULL duration_ms (Tier 1)."""
        row = MagicMock(
            state_id="state_1",
            token_id="token_1",
            node_id="node_1",
            status="pending",
            input_hash="hash_in",
            started_at=datetime.now(UTC),
            context_before_json=None,
            duration_ms=None,  # INVALID for PENDING
            output_hash=None,
            completed_at=None,
            context_after_json=None,
            error_json=None,
        )

        repo = NodeStateRepository(MagicMock())

        with pytest.raises(ValueError, match="PENDING.*duration_ms"):
            repo.load(row)

    def test_load_crashes_on_completed_without_output_hash(self) -> None:
        """Load crashes if COMPLETED state has NULL output_hash (Tier 1)."""
        row = MagicMock(
            state_id="state_1",
            token_id="token_1",
            node_id="node_1",
            status="completed",
            input_hash="hash_in",
            started_at=datetime.now(UTC),
            context_before_json=None,
            output_hash=None,  # INVALID for COMPLETED
            duration_ms=100,
            completed_at=datetime.now(UTC),
            context_after_json=None,
            error_json=None,
        )

        repo = NodeStateRepository(MagicMock())

        with pytest.raises(ValueError, match="COMPLETED.*output_hash"):
            repo.load(row)
```

**Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/core/landscape/test_node_state_repository.py -v`
Expected: FAIL with "cannot import name 'NodeStateRepository'"

**Step 3: Write minimal implementation**

Add to `repositories.py`:

```python
# Add imports at top of repositories.py
from elspeth.contracts.audit import (
    # ... existing imports ...
    NodeStateCompleted,
    NodeStateFailed,
    NodeStateOpen,
    NodeStatePending,
)
from elspeth.contracts.enums import (
    # ... existing imports ...
    NodeStateStatus,
)

# Type alias for the union
NodeState = NodeStateOpen | NodeStatePending | NodeStateCompleted | NodeStateFailed


class NodeStateRepository:
    """Repository for NodeState records (discriminated union).

    NodeState is a discriminated union with 4 variants based on status.
    This repository validates invariants and returns the correct type.

    Per Data Manifesto: Crashes on invalid data (Tier 1 trust).
    """

    def __init__(self, session: Any) -> None:
        self.session = session

    def load(self, row: Any) -> NodeState:
        """Load NodeState from database row.

        Returns the appropriate NodeState variant based on status.
        Validates required fields per status - crashes if missing.
        """
        status = NodeStateStatus(row.status)

        if status == NodeStateStatus.OPEN:
            return NodeStateOpen(
                state_id=row.state_id,
                token_id=row.token_id,
                node_id=row.node_id,
                status=status,
                input_hash=row.input_hash,
                started_at=row.started_at,
                context_before_json=row.context_before_json,
            )

        elif status == NodeStateStatus.PENDING:
            # PENDING requires duration_ms
            if row.duration_ms is None:
                raise ValueError(
                    f"PENDING state {row.state_id} has NULL duration_ms - "
                    "audit integrity violation"
                )
            return NodeStatePending(
                state_id=row.state_id,
                token_id=row.token_id,
                node_id=row.node_id,
                status=status,
                input_hash=row.input_hash,
                started_at=row.started_at,
                context_before_json=row.context_before_json,
                duration_ms=row.duration_ms,
            )

        elif status == NodeStateStatus.COMPLETED:
            # COMPLETED requires output_hash, duration_ms, completed_at
            if row.output_hash is None:
                raise ValueError(
                    f"COMPLETED state {row.state_id} has NULL output_hash - "
                    "audit integrity violation"
                )
            if row.duration_ms is None:
                raise ValueError(
                    f"COMPLETED state {row.state_id} has NULL duration_ms - "
                    "audit integrity violation"
                )
            if row.completed_at is None:
                raise ValueError(
                    f"COMPLETED state {row.state_id} has NULL completed_at - "
                    "audit integrity violation"
                )
            return NodeStateCompleted(
                state_id=row.state_id,
                token_id=row.token_id,
                node_id=row.node_id,
                status=status,
                input_hash=row.input_hash,
                started_at=row.started_at,
                context_before_json=row.context_before_json,
                output_hash=row.output_hash,
                duration_ms=row.duration_ms,
                completed_at=row.completed_at,
                context_after_json=row.context_after_json,
            )

        elif status == NodeStateStatus.FAILED:
            # FAILED requires duration_ms, completed_at
            if row.duration_ms is None:
                raise ValueError(
                    f"FAILED state {row.state_id} has NULL duration_ms - "
                    "audit integrity violation"
                )
            if row.completed_at is None:
                raise ValueError(
                    f"FAILED state {row.state_id} has NULL completed_at - "
                    "audit integrity violation"
                )
            return NodeStateFailed(
                state_id=row.state_id,
                token_id=row.token_id,
                node_id=row.node_id,
                status=status,
                input_hash=row.input_hash,
                started_at=row.started_at,
                context_before_json=row.context_before_json,
                duration_ms=row.duration_ms,
                completed_at=row.completed_at,
                error_json=row.error_json,
            )

        else:
            # Should never happen - enum exhaustiveness
            raise ValueError(f"Unknown NodeStateStatus: {status}")
```

**Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/core/landscape/test_node_state_repository.py -v`
Expected: PASS (6 tests)

**Step 5: Commit**

```bash
git add src/elspeth/core/landscape/repositories.py tests/core/landscape/test_node_state_repository.py
git commit -m "$(cat <<'EOF'
feat(landscape): add NodeStateRepository for discriminated union

Handles the 4-variant NodeState type with proper validation.
Crashes on invariant violations per Tier 1 trust model.
Extracted from recorder.py's _row_to_node_state().

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 1.4: Add Missing Error Repositories

**Files:**
- Modify: `src/elspeth/core/landscape/repositories.py`
- Test: `tests/core/landscape/test_error_repositories.py`

**Step 1: Write the failing test**

```python
# tests/core/landscape/test_error_repositories.py
"""Tests for error-related repositories."""

from datetime import UTC, datetime
from unittest.mock import MagicMock

import pytest

from elspeth.core.landscape.repositories import (
    ValidationErrorRepository,
    TransformErrorRepository,
    TokenOutcomeRepository,
)


class TestValidationErrorRepository:
    """Tests for ValidationErrorRepository."""

    def test_load_validation_error(self) -> None:
        """Load returns ValidationErrorRecord."""
        row = MagicMock(
            error_id="verr_abc123",
            run_id="run_1",
            node_id="source_1",
            row_hash="hash_abc",
            row_data_json='{"field": "value"}',
            error_json='{"code": "INVALID"}',
            schema_mode="strict",
            destination="quarantine",
            created_at=datetime.now(UTC),
        )

        repo = ValidationErrorRepository(MagicMock())
        result = repo.load(row)

        assert result.error_id == "verr_abc123"
        assert result.schema_mode == "strict"


class TestTransformErrorRepository:
    """Tests for TransformErrorRepository."""

    def test_load_transform_error(self) -> None:
        """Load returns TransformErrorRecord."""
        row = MagicMock(
            error_id="terr_abc123",
            state_id="state_1",
            token_id="token_1",
            error_type="validation",
            error_json='{"message": "failed"}',
            created_at=datetime.now(UTC),
        )

        repo = TransformErrorRepository(MagicMock())
        result = repo.load(row)

        assert result.error_id == "terr_abc123"
        assert result.error_type == "validation"


class TestTokenOutcomeRepository:
    """Tests for TokenOutcomeRepository."""

    def test_load_token_outcome(self) -> None:
        """Load returns TokenOutcome with enum conversion."""
        row = MagicMock(
            token_id="token_1",
            outcome="completed",
            sink_id="output",
            recorded_at=datetime.now(UTC),
        )

        repo = TokenOutcomeRepository(MagicMock())
        result = repo.load(row)

        assert result.token_id == "token_1"
        # outcome should be converted to TokenOutcomeType enum
        from elspeth.contracts.enums import TokenOutcomeType
        assert result.outcome == TokenOutcomeType.COMPLETED
```

**Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/core/landscape/test_error_repositories.py -v`
Expected: FAIL with import errors

**Step 3: Write minimal implementation**

Add to `repositories.py`:

```python
# Add to imports
from elspeth.contracts.audit import (
    # ... existing ...
    ValidationErrorRecord,
    TransformErrorRecord,
    TokenOutcome,
)
from elspeth.contracts.enums import (
    # ... existing ...
    TokenOutcomeType,
)


class ValidationErrorRepository:
    """Repository for ValidationErrorRecord."""

    def __init__(self, session: Any) -> None:
        self.session = session

    def load(self, row: Any) -> ValidationErrorRecord:
        """Load ValidationErrorRecord from database row."""
        return ValidationErrorRecord(
            error_id=row.error_id,
            run_id=row.run_id,
            node_id=row.node_id,
            row_hash=row.row_hash,
            row_data_json=row.row_data_json,
            error_json=row.error_json,
            schema_mode=row.schema_mode,
            destination=row.destination,
            created_at=row.created_at,
        )


class TransformErrorRepository:
    """Repository for TransformErrorRecord."""

    def __init__(self, session: Any) -> None:
        self.session = session

    def load(self, row: Any) -> TransformErrorRecord:
        """Load TransformErrorRecord from database row."""
        return TransformErrorRecord(
            error_id=row.error_id,
            state_id=row.state_id,
            token_id=row.token_id,
            error_type=row.error_type,
            error_json=row.error_json,
            created_at=row.created_at,
        )


class TokenOutcomeRepository:
    """Repository for TokenOutcome records."""

    def __init__(self, session: Any) -> None:
        self.session = session

    def load(self, row: Any) -> TokenOutcome:
        """Load TokenOutcome from database row."""
        return TokenOutcome(
            token_id=row.token_id,
            outcome=TokenOutcomeType(row.outcome),  # Enum conversion
            sink_id=row.sink_id,
            recorded_at=row.recorded_at,
        )
```

**Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/core/landscape/test_error_repositories.py -v`
Expected: PASS (3 tests)

**Step 5: Commit**

```bash
git add src/elspeth/core/landscape/repositories.py tests/core/landscape/test_error_repositories.py
git commit -m "$(cat <<'EOF'
feat(landscape): add error/outcome repositories

Adds ValidationErrorRepository, TransformErrorRepository, and
TokenOutcomeRepository for audit error tracking.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 1.5: Add Artifact and BatchMember Repositories

**Files:**
- Modify: `src/elspeth/core/landscape/repositories.py`
- Test: `tests/core/landscape/test_artifact_repository.py`

**Step 1: Write the failing test**

```python
# tests/core/landscape/test_artifact_repository.py
"""Tests for ArtifactRepository and BatchMemberRepository."""

from datetime import UTC, datetime
from unittest.mock import MagicMock

from elspeth.core.landscape.repositories import (
    ArtifactRepository,
    BatchMemberRepository,
)


class TestArtifactRepository:
    """Tests for ArtifactRepository."""

    def test_load_artifact(self) -> None:
        """Load returns Artifact."""
        row = MagicMock(
            artifact_id="art_123",
            run_id="run_1",
            sink_id="output",
            token_id="token_1",
            content_hash="hash_abc",
            content_ref="ref_abc",
            created_at=datetime.now(UTC),
        )

        repo = ArtifactRepository(MagicMock())
        result = repo.load(row)

        assert result.artifact_id == "art_123"
        assert result.content_hash == "hash_abc"


class TestBatchMemberRepository:
    """Tests for BatchMemberRepository."""

    def test_load_batch_member(self) -> None:
        """Load returns BatchMember."""
        row = MagicMock(
            batch_id="batch_1",
            token_id="token_1",
            ordinal=0,
            added_at=datetime.now(UTC),
        )

        repo = BatchMemberRepository(MagicMock())
        result = repo.load(row)

        assert result.batch_id == "batch_1"
        assert result.ordinal == 0
```

**Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/core/landscape/test_artifact_repository.py -v`
Expected: FAIL with import errors

**Step 3: Write minimal implementation**

Add to `repositories.py`:

```python
# Add to imports
from elspeth.contracts.audit import (
    # ... existing ...
    Artifact,
    BatchMember,
)


class ArtifactRepository:
    """Repository for Artifact records."""

    def __init__(self, session: Any) -> None:
        self.session = session

    def load(self, row: Any) -> Artifact:
        """Load Artifact from database row."""
        return Artifact(
            artifact_id=row.artifact_id,
            run_id=row.run_id,
            sink_id=row.sink_id,
            token_id=row.token_id,
            content_hash=row.content_hash,
            content_ref=row.content_ref,
            created_at=row.created_at,
        )


class BatchMemberRepository:
    """Repository for BatchMember records."""

    def __init__(self, session: Any) -> None:
        self.session = session

    def load(self, row: Any) -> BatchMember:
        """Load BatchMember from database row."""
        return BatchMember(
            batch_id=row.batch_id,
            token_id=row.token_id,
            ordinal=row.ordinal,
            added_at=row.added_at,
        )
```

**Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/core/landscape/test_artifact_repository.py -v`
Expected: PASS (2 tests)

**Step 5: Commit**

```bash
git add src/elspeth/core/landscape/repositories.py tests/core/landscape/test_artifact_repository.py
git commit -m "$(cat <<'EOF'
feat(landscape): add Artifact and BatchMember repositories

Completes the repository layer for all audit record types.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Phase 2: Integration (Medium Risk)

### Task 2.1: Integrate Repositories into Recorder - Runs

**Files:**
- Modify: `src/elspeth/core/landscape/recorder.py`
- Test: Run existing tests

**Step 1: Add repository imports to recorder.py**

At top of file, add:

```python
from elspeth.core.landscape.repositories import (
    RunRepository,
    NodeRepository,
    EdgeRepository,
    RowRepository,
    TokenRepository,
    TokenParentRepository,
    CallRepository,
    RoutingEventRepository,
    BatchRepository,
    NodeStateRepository,
    ValidationErrorRepository,
    TransformErrorRepository,
    TokenOutcomeRepository,
    ArtifactRepository,
    BatchMemberRepository,
)
from elspeth.core.landscape._database_ops import DatabaseOps
from elspeth.core.landscape._helpers import now, generate_id, coerce_enum
```

**Step 2: Initialize repositories in __init__**

```python
def __init__(self, db: LandscapeDB, *, payload_store: Any | None = None) -> None:
    self._db = db
    self._ops = DatabaseOps(db)  # NEW
    self._payload_store = payload_store
    self._call_indices: dict[str, int] = {}
    self._call_index_lock: Lock = Lock()

    # Initialize repositories (lazy - session passed per-call)
    self._run_repo = RunRepository(None)
    self._node_repo = NodeRepository(None)
    self._edge_repo = EdgeRepository(None)
    # ... etc
```

**Step 3: Replace inline get_run() with repository**

Find and replace `get_run()` method to use repository:

```python
def get_run(self, run_id: str) -> Run | None:
    """Get run by ID."""
    query = select(runs_table).where(runs_table.c.run_id == run_id)
    row = self._ops.execute_fetchone(query)
    if row is None:
        return None
    return self._run_repo.load(row)
```

**Step 4: Run existing tests**

Run: `.venv/bin/python -m pytest tests/core/landscape/ -v -x`
Expected: PASS (all existing tests should still pass)

**Step 5: Commit**

```bash
git add src/elspeth/core/landscape/recorder.py
git commit -m "$(cat <<'EOF'
refactor(landscape): integrate RunRepository into recorder

First step of repository integration. Uses DatabaseOps helper
and RunRepository.load() for get_run(). No behavior change.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 2.2-2.9: Continue Repository Integration

Follow the same pattern for each domain:

| Task | Repository | Methods to Refactor |
|------|------------|---------------------|
| 2.2 | NodeRepository | `get_node()`, `get_nodes()`, `_row_to_node()` |
| 2.3 | EdgeRepository | `get_edges()`, `get_edge_map()` |
| 2.4 | RowRepository | `get_row()`, `get_rows()` |
| 2.5 | TokenRepository | `get_token()`, `get_tokens()` |
| 2.6 | NodeStateRepository | `get_node_state()`, `get_node_states_for_token()`, remove `_row_to_node_state()` |
| 2.7 | CallRepository | `get_calls()`, `find_call_by_request_hash()` |
| 2.8 | BatchRepository | `get_batch()`, `get_batches()`, `get_batch_members()` |
| 2.9 | Error/Outcome Repos | All error query methods |

Each follows TDD cycle: test → implement → test → commit.

---

## Phase 3: Facade Pattern (Lower Risk)

### Task 3.1: Create Repository Container

**Files:**
- Create: `src/elspeth/core/landscape/repository_container.py`
- Test: `tests/core/landscape/test_repository_container.py`

**Implementation:**

```python
# src/elspeth/core/landscape/repository_container.py
"""Container for all landscape repositories.

Provides centralized access to repositories with lazy initialization.
Used by LandscapeRecorder facade.
"""

from typing import TYPE_CHECKING

from elspeth.core.landscape.repositories import (
    ArtifactRepository,
    BatchMemberRepository,
    BatchRepository,
    CallRepository,
    EdgeRepository,
    NodeRepository,
    NodeStateRepository,
    RoutingEventRepository,
    RowRepository,
    RunRepository,
    TokenOutcomeRepository,
    TokenParentRepository,
    TokenRepository,
    TransformErrorRepository,
    ValidationErrorRepository,
)

if TYPE_CHECKING:
    from elspeth.core.landscape.database import LandscapeDB


class RepositoryContainer:
    """Container providing access to all landscape repositories."""

    def __init__(self, db: "LandscapeDB") -> None:
        self._db = db

        # Initialize all repositories
        self.runs = RunRepository(None)
        self.nodes = NodeRepository(None)
        self.edges = EdgeRepository(None)
        self.rows = RowRepository(None)
        self.tokens = TokenRepository(None)
        self.token_parents = TokenParentRepository(None)
        self.node_states = NodeStateRepository(None)
        self.calls = CallRepository(None)
        self.routing_events = RoutingEventRepository(None)
        self.batches = BatchRepository(None)
        self.batch_members = BatchMemberRepository(None)
        self.validation_errors = ValidationErrorRepository(None)
        self.transform_errors = TransformErrorRepository(None)
        self.token_outcomes = TokenOutcomeRepository(None)
        self.artifacts = ArtifactRepository(None)
```

---

## Phase 4: Final Cleanup

### Task 4.1: Remove Duplicate Helper Functions

Remove from recorder.py:
- `_now()` → use `from ._helpers import now`
- `_generate_id()` → use `from ._helpers import generate_id`
- `_coerce_enum()` → use `from ._helpers import coerce_enum`
- `_row_to_node_state()` → use `NodeStateRepository.load()`
- `_row_to_node()` → use `NodeRepository.load()`

### Task 4.2: Replace Connection Blocks

Search and replace pattern:
```python
# Before
with self._db.connection() as conn:
    result = conn.execute(query)
    row = result.fetchone()

# After
row = self._ops.execute_fetchone(query)
```

### Task 4.3: Final Test Suite

Run: `.venv/bin/python -m pytest tests/ -v --tb=short`
Expected: All tests pass

---

## Verification Checklist

After each phase:
- [ ] All existing tests pass
- [ ] New tests have >80% coverage of new code
- [ ] No new mypy errors
- [ ] No new ruff warnings
- [ ] recorder.py line count reduced

Final verification:
- [ ] recorder.py < 500 lines
- [ ] repositories.py < 700 lines
- [ ] All audit integrity tests pass
- [ ] Integration tests pass
