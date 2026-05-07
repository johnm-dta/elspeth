# Web UX Task-Plan 2C: Session Data Layer

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans.

**Goal:** Define SQLAlchemy Core table models and SessionService protocol with record types
**Parent Plan:** `plans/2026-03-28-web-ux-sub2-auth-sessions.md`
**Spec:** `specs/2026-03-28-web-ux-sub2-auth-sessions-design.md`
**Depends On:** Sub-Plan 1 (Foundation) -- completed
**Blocks:** Task-Plan 2D (SessionServiceImpl)

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| Create | `src/elspeth/web/sessions/__init__.py` | Module init |
| Create | `src/elspeth/web/sessions/models.py` | SQLAlchemy Core table definitions (sessions, chat_messages, composition_states, runs, run_events) |
| Create | `src/elspeth/web/sessions/protocol.py` | SessionServiceProtocol, record dataclasses, RunAlreadyActiveError |
| Create | `tests/unit/web/sessions/__init__.py` | Test package |
| Create | `tests/unit/web/sessions/test_models.py` | Table schema tests |
| Create | `tests/unit/web/sessions/test_protocol.py` | Protocol and record type tests |

---

### Task 2.7: Session Database Models

**Files:**
- Create: `src/elspeth/web/sessions/__init__.py`
- Create: `src/elspeth/web/sessions/models.py`
- Create: `tests/unit/web/sessions/__init__.py`
- Create: `tests/unit/web/sessions/test_models.py`

- [ ] **Step 1: Write tests**

```python
# tests/unit/web/sessions/__init__.py
```

```python
# tests/unit/web/sessions/test_models.py
"""Tests for SQLAlchemy session table definitions."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pytest
from sqlalchemy import create_engine, inspect, select, insert
from sqlalchemy.exc import IntegrityError

from elspeth.web.sessions.models import (
    metadata,
    sessions_table,
    chat_messages_table,
    composition_states_table,
    runs_table,
    run_events_table,
)


@pytest.fixture
def engine():
    """Create an in-memory SQLite engine with all tables."""
    eng = create_engine("sqlite:///:memory:")
    metadata.create_all(eng)
    return eng


class TestTableCreation:
    """Verify all five tables are created with correct schemas."""

    def test_all_tables_exist(self, engine) -> None:
        inspector = inspect(engine)
        table_names = inspector.get_table_names()
        assert "sessions" in table_names
        assert "chat_messages" in table_names
        assert "composition_states" in table_names
        assert "runs" in table_names
        assert "run_events" in table_names

    def test_sessions_columns(self, engine) -> None:
        inspector = inspect(engine)
        columns = {c["name"] for c in inspector.get_columns("sessions")}
        assert columns >= {"id", "user_id", "title", "created_at", "updated_at"}

    def test_chat_messages_columns(self, engine) -> None:
        inspector = inspect(engine)
        columns = {c["name"] for c in inspector.get_columns("chat_messages")}
        assert columns >= {
            "id", "session_id", "role", "content", "tool_calls", "created_at",
        }

    def test_composition_states_columns(self, engine) -> None:
        inspector = inspect(engine)
        columns = {c["name"] for c in inspector.get_columns("composition_states")}
        assert columns >= {
            "id", "session_id", "version", "source", "nodes", "edges",
            "outputs", "metadata_", "is_valid", "validation_errors", "created_at",
        }

    def test_runs_columns(self, engine) -> None:
        inspector = inspect(engine)
        columns = {c["name"] for c in inspector.get_columns("runs")}
        assert columns >= {
            "id", "session_id", "state_id", "status", "started_at",
            "finished_at", "rows_processed", "rows_failed", "error",
            "landscape_run_id", "pipeline_yaml",
        }

    def test_run_events_columns(self, engine) -> None:
        inspector = inspect(engine)
        columns = {c["name"] for c in inspector.get_columns("run_events")}
        assert columns >= {"id", "run_id", "timestamp", "event_type", "data"}


class TestCompositionStateUniqueConstraint:
    """Verify the UNIQUE(session_id, version) constraint."""

    def test_duplicate_version_raises(self, engine) -> None:
        session_id = str(uuid.uuid4())
        state_id_1 = str(uuid.uuid4())
        state_id_2 = str(uuid.uuid4())

        with engine.begin() as conn:
            # Insert a session first (FK constraint)
            conn.execute(
                insert(sessions_table).values(
                    id=session_id,
                    user_id="alice",
                    title="Test",
                    created_at=datetime.now(timezone.utc),
                    updated_at=datetime.now(timezone.utc),
                )
            )
            # First state version
            conn.execute(
                insert(composition_states_table).values(
                    id=state_id_1,
                    session_id=session_id,
                    version=1,
                    is_valid=False,
                    created_at=datetime.now(timezone.utc),
                )
            )
            # Duplicate version should fail
            with pytest.raises(IntegrityError):
                conn.execute(
                    insert(composition_states_table).values(
                        id=state_id_2,
                        session_id=session_id,
                        version=1,
                        is_valid=False,
                        created_at=datetime.now(timezone.utc),
                    )
                )


class TestSessionForeignKeys:
    """Verify foreign key relationships."""

    def test_chat_message_requires_valid_session(self, engine) -> None:
        """Inserting a message with a nonexistent session_id should fail
        if FK enforcement is on (SQLite needs PRAGMA foreign_keys=ON)."""
        # SQLite does not enforce FK by default; this test verifies
        # the column exists and accepts valid references.
        session_id = str(uuid.uuid4())
        msg_id = str(uuid.uuid4())

        with engine.begin() as conn:
            # Enable FK enforcement for SQLite
            conn.execute(insert(sessions_table).values(
                id=session_id,
                user_id="alice",
                title="Test",
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
            ))
            conn.execute(insert(chat_messages_table).values(
                id=msg_id,
                session_id=session_id,
                role="user",
                content="Hello",
                created_at=datetime.now(timezone.utc),
            ))
            # Verify it was inserted
            result = conn.execute(
                select(chat_messages_table).where(
                    chat_messages_table.c.id == msg_id
                )
            ).fetchone()
            assert result is not None
```

- [ ] **Step 2: Run tests -- verify they fail**

```bash
.venv/bin/python -m pytest tests/unit/web/sessions/test_models.py -v
```

Expected: `ModuleNotFoundError: No module named 'elspeth.web.sessions'`

- [ ] **Step 3: Implement SQLAlchemy table models**

```python
# src/elspeth/web/sessions/__init__.py
"""Session management -- persistence, CRUD, and API routes."""
```

```python
# src/elspeth/web/sessions/models.py
"""SQLAlchemy Core table definitions for the session database.

Tables: sessions, chat_messages, composition_states, runs, run_events.
Schema creation via metadata.create_all(engine) on startup.

All tables live in a dedicated session database, separate from the
Landscape audit database.
"""

from __future__ import annotations

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Column,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    UniqueConstraint,
)
from sqlalchemy.types import JSON

metadata = MetaData()

sessions_table = Table(
    "sessions",
    metadata,
    Column("id", String, primary_key=True),
    Column("user_id", String, nullable=False, index=True),
    Column("title", String, nullable=False),
    Column("created_at", DateTime, nullable=False),
    Column("updated_at", DateTime, nullable=False),
)

chat_messages_table = Table(
    "chat_messages",
    metadata,
    Column("id", String, primary_key=True),
    Column(
        "session_id", String,
        ForeignKey("sessions.id", ondelete="CASCADE"),
        nullable=False, index=True,
    ),
    Column("role", String, nullable=False),
    Column("content", Text, nullable=False),
    Column("tool_calls", JSON, nullable=True),
    Column("created_at", DateTime, nullable=False),
    CheckConstraint(
        "role IN ('user', 'assistant', 'system', 'tool')",
        name="ck_chat_messages_role",
    ),
)

composition_states_table = Table(
    "composition_states",
    metadata,
    Column("id", String, primary_key=True),
    Column(
        "session_id", String,
        ForeignKey("sessions.id", ondelete="CASCADE"),
        nullable=False, index=True,
    ),
    Column("version", Integer, nullable=False),
    Column("source", JSON, nullable=True),
    Column("nodes", JSON, nullable=True),
    Column("edges", JSON, nullable=True),
    Column("outputs", JSON, nullable=True),
    Column("metadata_", JSON, nullable=True),
    Column("is_valid", Boolean, nullable=False, default=False),
    Column("validation_errors", JSON, nullable=True),
    Column("created_at", DateTime, nullable=False),
    UniqueConstraint("session_id", "version", name="uq_composition_state_version"),
)

runs_table = Table(
    "runs",
    metadata,
    Column("id", String, primary_key=True),
    Column(
        "session_id", String,
        ForeignKey("sessions.id", ondelete="CASCADE"),
        nullable=False, index=True,
    ),
    Column(
        "state_id", String,
        ForeignKey("composition_states.id"),
        nullable=False,
    ),
    Column("status", String, nullable=False),
    Column("started_at", DateTime, nullable=False),
    Column("finished_at", DateTime, nullable=True),
    Column("rows_processed", Integer, nullable=False, default=0),
    Column("rows_failed", Integer, nullable=False, default=0),
    Column("error", Text, nullable=True),
    Column("landscape_run_id", String, nullable=True),
    Column("pipeline_yaml", Text, nullable=True),
    CheckConstraint(
        "status IN ('pending', 'running', 'completed', 'failed', 'cancelled')",
        name="ck_runs_status",
    ),
)

run_events_table = Table(
    "run_events",
    metadata,
    Column("id", String, primary_key=True),
    Column(
        "run_id", String,
        ForeignKey("runs.id", ondelete="CASCADE"),
        nullable=False, index=True,
    ),
    Column("timestamp", DateTime, nullable=False),
    Column("event_type", String, nullable=False),
    Column("data", JSON, nullable=False),
    CheckConstraint(
        "event_type IN ('progress', 'error', 'completed')",
        name="ck_run_events_type",
    ),
)
```

- [ ] **Step 4: Run tests -- verify they pass**

```bash
.venv/bin/python -m pytest tests/unit/web/sessions/test_models.py -v
```

Expected: all 9 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/elspeth/web/sessions/__init__.py src/elspeth/web/sessions/models.py \
    tests/unit/web/sessions/__init__.py tests/unit/web/sessions/test_models.py
git commit -m "feat(web/sessions): add SQLAlchemy Core table definitions for session database"
```

---

### Task 2.8: SessionService Protocol and Record Types

**Files:**
- Create: `src/elspeth/web/sessions/protocol.py`
- Create: `tests/unit/web/sessions/test_protocol.py`

- [ ] **Step 1: Write tests**

```python
# tests/unit/web/sessions/test_protocol.py
"""Tests for session record dataclasses and protocol definition."""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

import pytest

from elspeth.web.sessions.protocol import (
    ChatMessageRecord,
    CompositionStateData,
    CompositionStateRecord,
    RunAlreadyActiveError,
    RunRecord,
    SessionRecord,
    SessionServiceProtocol,
)


class TestSessionRecord:
    def test_frozen_immutability(self) -> None:
        record = SessionRecord(
            id=uuid4(), user_id="alice", title="Test",
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        with pytest.raises(AttributeError):
            record.title = "Changed"  # type: ignore[misc]


class TestChatMessageRecord:
    def test_tool_calls_frozen_when_present(self) -> None:
        record = ChatMessageRecord(
            id=uuid4(), session_id=uuid4(), role="assistant",
            content="Hello", tool_calls={"name": "search", "args": {"q": "test"}},
            created_at=datetime.now(timezone.utc),
        )
        with pytest.raises(TypeError):
            record.tool_calls["new_key"] = "value"  # type: ignore[index]

    def test_tool_calls_none_is_fine(self) -> None:
        record = ChatMessageRecord(
            id=uuid4(), session_id=uuid4(), role="user",
            content="Hello", tool_calls=None,
            created_at=datetime.now(timezone.utc),
        )
        assert record.tool_calls is None


class TestCompositionStateData:
    def test_mutable_inputs_are_frozen(self) -> None:
        source = {"type": "csv", "path": "/data/test.csv"}
        nodes = [{"id": "n1", "type": "source"}]
        data = CompositionStateData(
            source=source, nodes=nodes, is_valid=True,
        )
        # Original dicts should not affect the frozen copy
        source["type"] = "json"
        assert data.source["type"] == "csv"  # type: ignore[index]
        # Frozen containers should reject mutation
        with pytest.raises(TypeError):
            data.source["new_key"] = "value"  # type: ignore[index]
        with pytest.raises(TypeError):
            data.nodes.append({"id": "n2"})  # type: ignore[union-attr]

    def test_none_fields_not_frozen(self) -> None:
        data = CompositionStateData(is_valid=False)
        assert data.source is None
        assert data.nodes is None

    def test_frozen_immutability(self) -> None:
        data = CompositionStateData(is_valid=True)
        with pytest.raises(AttributeError):
            data.is_valid = False  # type: ignore[misc]


class TestCompositionStateRecord:
    def test_mutable_fields_are_frozen(self) -> None:
        record = CompositionStateRecord(
            id=uuid4(), session_id=uuid4(), version=1,
            source={"type": "csv"}, nodes=[{"id": "n1"}],
            edges=None, outputs=None, metadata_=None,
            is_valid=True, validation_errors=None,
            created_at=datetime.now(timezone.utc),
        )
        with pytest.raises(TypeError):
            record.source["new"] = "value"  # type: ignore[index]


class TestRunRecord:
    def test_frozen_immutability(self) -> None:
        record = RunRecord(
            id=uuid4(), session_id=uuid4(), state_id=uuid4(),
            status="running", started_at=datetime.now(timezone.utc),
            finished_at=None, rows_processed=0, rows_failed=0,
            error=None, landscape_run_id=None, pipeline_yaml=None,
        )
        with pytest.raises(AttributeError):
            record.status = "completed"  # type: ignore[misc]


class TestRunAlreadyActiveError:
    def test_construction_and_message(self) -> None:
        err = RunAlreadyActiveError("session-123")
        assert err.session_id == "session-123"
        assert "session-123" in str(err)
        assert isinstance(err, Exception)


class TestSessionServiceProtocol:
    def test_is_runtime_checkable(self) -> None:
        assert hasattr(SessionServiceProtocol, "__protocol_attrs__") or hasattr(
            SessionServiceProtocol, "_is_runtime_protocol"
        )
```

- [ ] **Step 2: Implement protocol and record dataclasses**

```python
# src/elspeth/web/sessions/protocol.py
"""SessionService protocol and record dataclasses.

Record types are frozen dataclasses representing database rows.
CompositionStateData is the input DTO for saving new state versions.
"""

# ID Convention: Record dataclasses use UUID for type safety. The database
# stores IDs as String (TEXT). The SessionServiceImpl (Sub-2d) converts
# between UUID and str at the query/record boundary. Callers work with
# UUID exclusively; the storage representation is an implementation detail.

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol, runtime_checkable
from uuid import UUID

from elspeth.contracts.freeze import freeze_fields


@dataclass(frozen=True, slots=True)
class SessionRecord:
    """Represents a row from the sessions table.

    All fields are scalars or datetime -- no freeze guard needed.
    """

    id: UUID
    user_id: str
    title: str
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class ChatMessageRecord:
    """Represents a row from the chat_messages table.

    tool_calls may contain a dict -- requires freeze guard when not None.
    """

    id: UUID
    session_id: UUID
    role: str
    content: str
    tool_calls: Mapping[str, Any] | None
    created_at: datetime

    def __post_init__(self) -> None:
        if self.tool_calls is not None:
            freeze_fields(self, "tool_calls")


@dataclass(frozen=True, slots=True)
class CompositionStateData:
    """Input DTO for saving a new composition state version.

    Contains mutable container fields -- requires freeze guard.
    """

    source: dict[str, Any] | None = None
    nodes: list[dict[str, Any]] | None = None
    edges: list[dict[str, Any]] | None = None
    outputs: list[dict[str, Any]] | None = None
    metadata_: dict[str, Any] | None = None
    is_valid: bool = False
    validation_errors: list[str] | None = None

    def __post_init__(self) -> None:
        non_none = []
        if self.source is not None:
            non_none.append("source")
        if self.nodes is not None:
            non_none.append("nodes")
        if self.edges is not None:
            non_none.append("edges")
        if self.outputs is not None:
            non_none.append("outputs")
        if self.metadata_ is not None:
            non_none.append("metadata_")
        if self.validation_errors is not None:
            non_none.append("validation_errors")
        if non_none:
            freeze_fields(self, *non_none)


@dataclass(frozen=True, slots=True)
class CompositionStateRecord:
    """Represents a row from the composition_states table.

    Contains mutable container fields -- requires freeze guard.
    """

    id: UUID
    session_id: UUID
    version: int
    source: Mapping[str, Any] | None
    nodes: Sequence[Mapping[str, Any]] | None
    edges: Sequence[Mapping[str, Any]] | None
    outputs: Sequence[Mapping[str, Any]] | None
    metadata_: Mapping[str, Any] | None
    is_valid: bool
    validation_errors: Sequence[str] | None
    created_at: datetime

    def __post_init__(self) -> None:
        non_none = []
        if self.source is not None:
            non_none.append("source")
        if self.nodes is not None:
            non_none.append("nodes")
        if self.edges is not None:
            non_none.append("edges")
        if self.outputs is not None:
            non_none.append("outputs")
        if self.metadata_ is not None:
            non_none.append("metadata_")
        if self.validation_errors is not None:
            non_none.append("validation_errors")
        if non_none:
            freeze_fields(self, *non_none)


@dataclass(frozen=True, slots=True)
class RunRecord:
    """Represents a row from the runs table.

    All fields are scalars, datetime, or None -- no freeze guard needed.
    """

    id: UUID
    session_id: UUID
    state_id: UUID
    status: str
    started_at: datetime
    finished_at: datetime | None
    rows_processed: int
    rows_failed: int
    error: str | None
    landscape_run_id: str | None
    pipeline_yaml: str | None


class RunAlreadyActiveError(Exception):
    """Raised when attempting to create a run while one is already active.

    Seam contract D: HTTP handlers catching this error MUST return 409 with
    {"detail": str(exc), "error_type": "run_already_active"} -- not a bare
    HTTPException. See seam-contracts.md for the canonical error shape.
    """

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        super().__init__(
            f"Session {session_id} already has an active run"
        )


@runtime_checkable
class SessionServiceProtocol(Protocol):
    """Protocol for session persistence operations."""

    async def create_session(
        self, user_id: str, title: str,
    ) -> SessionRecord: ...

    async def get_session(self, session_id: UUID) -> SessionRecord: ...

    async def list_sessions(self, user_id: str) -> list[SessionRecord]: ...

    async def archive_session(self, session_id: UUID) -> None: ...

    async def add_message(
        self,
        session_id: UUID,
        role: str,
        content: str,
        tool_calls: dict[str, Any] | None = None,
    ) -> ChatMessageRecord: ...

    async def get_messages(
        self, session_id: UUID,
    ) -> list[ChatMessageRecord]: ...

    async def save_composition_state(
        self, session_id: UUID, state: CompositionStateData,
    ) -> CompositionStateRecord: ...

    async def get_current_state(
        self, session_id: UUID,
    ) -> CompositionStateRecord | None: ...

    async def get_state(self, state_id: UUID) -> CompositionStateRecord: ...

    async def get_state_versions(
        self, session_id: UUID,
    ) -> list[CompositionStateRecord]: ...

    async def set_active_state(
        self, session_id: UUID, state_id: UUID,
    ) -> CompositionStateRecord: ...

    async def create_run(
        self,
        session_id: UUID,
        state_id: UUID,
        pipeline_yaml: str | None = None,
    ) -> RunRecord: ...

    async def get_run(self, run_id: UUID) -> RunRecord: ...

    async def update_run_status(
        self,
        run_id: UUID,
        status: str,
        error: str | None = None,
        landscape_run_id: str | None = None,
        rows_processed: int | None = None,
        rows_failed: int | None = None,
    ) -> None: ...

    async def get_active_run(
        self, session_id: UUID,
    ) -> RunRecord | None: ...
```

- [ ] **Step 3: Commit**

```bash
git add src/elspeth/web/sessions/protocol.py \
    tests/unit/web/sessions/test_protocol.py
git commit -m "feat(web/sessions): add SessionServiceProtocol, record types, and RunAlreadyActiveError"
```

---

## Self-Review Checklist

- [ ] Both tasks (2.7 and 2.8) complete with all steps checked
- [ ] All 5 tables created: sessions, chat_messages, composition_states, runs, run_events
- [ ] UNIQUE(session_id, version) constraint on composition_states
- [ ] CHECK constraints on role, status, and event_type columns
- [ ] Foreign keys with CASCADE deletes where appropriate
- [ ] All 9 table schema tests pass (test_models.py)
- [ ] All 9 protocol/record tests pass (test_protocol.py)
- [ ] Record dataclasses use `frozen=True, slots=True`
- [ ] Container fields in records use `freeze_fields()` in `__post_init__`
- [ ] Scalar-only records (SessionRecord, RunRecord) have no freeze guard
- [ ] `CompositionStateData` DTO freezes all non-None container fields
- [ ] `RunAlreadyActiveError` documents seam contract D (409 response shape)
- [ ] `SessionServiceProtocol` is `@runtime_checkable`
- [ ] No imports from `elspeth.web.auth` -- session data layer is independent
- [ ] All commits use conventional commit format
