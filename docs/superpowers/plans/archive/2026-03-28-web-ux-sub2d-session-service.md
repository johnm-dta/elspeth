# Web UX Task-Plan 2D: SessionServiceImpl

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans.

**Goal:** Implement SessionServiceImpl -- full CRUD, composition state versioning, Run enforcement, revert, state hydration
**Parent Plan:** `plans/2026-03-28-web-ux-sub2-auth-sessions.md`
**Spec:** `specs/2026-03-28-web-ux-sub2-auth-sessions-design.md`
**Depends On:** Task-Plan 2C (Session Data Layer)
**Blocks:** Task-Plan 2E (Session API & Wiring)

---

## Post-2C Alignment Notes

This plan was updated after Sub-2C completion to align with the actual protocol
and models. Changes from the original plan:

| # | Issue | Resolution |
|---|-------|------------|
| D1 | `SessionRecord` gained `auth_provider_type` field | Added `auth_provider_type` param to `create_session` calls and record constructions |
| D2 | `CompositionStateRecord` has `derived_from_state_id` field | Added to all record constructions; `None` for new states, source UUID for reverts |
| D3 | `LEGAL_RUN_TRANSITIONS` not enforced | Added transition validation in `update_run_status` with tests |
| D4 | `landscape_run_id` write-once semantics | Added guard in `update_run_status`, test for double-write rejection |
| D5 | `cancel_orphaned_runs` missing | Added implementation and tests |
| D6 | `set_active_state` didn't record lineage | `derived_from_state_id` set to source state UUID, INSERT includes column |
| D7 | `save_composition_state` INSERT missing `derived_from_state_id` | Explicit `derived_from_state_id=None` in INSERT |

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| Create | `src/elspeth/web/sessions/service.py` | SessionServiceImpl -- CRUD, state versioning, active run check |
| Create | `tests/unit/web/sessions/test_service.py` | SessionService CRUD tests |

---

## Pre-requisites

Task-Plan 2C (Session Data Layer) must be complete. The following files must exist:

- `src/elspeth/web/sessions/__init__.py`
- `src/elspeth/web/sessions/protocol.py` (SessionServiceProtocol, record types, CompositionStateData, RunAlreadyActiveError)
- `src/elspeth/web/sessions/models.py` (SQLAlchemy Core table definitions: sessions, chat_messages, composition_states, runs, run_events)

---

### Task 2.9: SessionServiceImpl

**Files:**
- Create: `src/elspeth/web/sessions/service.py`
- Create: `tests/unit/web/sessions/test_service.py`

- [ ] **Step 1: Write tests**

```python
# tests/unit/web/sessions/test_service.py
"""Tests for SessionServiceImpl -- CRUD, state versioning, active run enforcement."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pytest
from sqlalchemy import create_engine

from elspeth.web.sessions.models import (
    metadata,
    runs_table,
    composition_states_table,
)
from elspeth.web.sessions.protocol import (
    LEGAL_RUN_TRANSITIONS,
    CompositionStateData,
    RunAlreadyActiveError,
    RunRecord,
    SessionRecord,
    ChatMessageRecord,
    CompositionStateRecord,
)
from elspeth.web.sessions.service import SessionServiceImpl


@pytest.fixture
def engine():
    """Create an in-memory SQLite engine with all tables."""
    eng = create_engine("sqlite:///:memory:")
    metadata.create_all(eng)
    return eng


@pytest.fixture
def service(engine):
    """Create a SessionServiceImpl backed by the in-memory engine."""
    return SessionServiceImpl(engine)


class TestSessionCRUD:
    """Tests for session create, get, list, and archive."""

    @pytest.mark.asyncio
    async def test_create_session(self, service) -> None:
        session = await service.create_session("alice", "My Session", "local")
        assert isinstance(session, SessionRecord)
        assert session.user_id == "alice"
        assert session.auth_provider_type == "local"
        assert session.title == "My Session"
        assert isinstance(session.id, uuid.UUID)
        assert isinstance(session.created_at, datetime)

    @pytest.mark.asyncio
    async def test_get_session(self, service) -> None:
        created = await service.create_session("alice", "Test", "local")
        fetched = await service.get_session(created.id)
        assert fetched.id == created.id
        assert fetched.user_id == "alice"
        assert fetched.title == "Test"

    @pytest.mark.asyncio
    async def test_get_session_not_found_raises(self, service) -> None:
        with pytest.raises(ValueError, match="not found"):
            await service.get_session(uuid.uuid4())

    @pytest.mark.asyncio
    async def test_list_sessions_user_scoped(self, service) -> None:
        await service.create_session("alice", "Session A", "local")
        await service.create_session("alice", "Session B", "local")
        await service.create_session("bob", "Session C", "local")

        alice_sessions = await service.list_sessions("alice")
        assert len(alice_sessions) == 2
        assert all(s.user_id == "alice" for s in alice_sessions)

        bob_sessions = await service.list_sessions("bob")
        assert len(bob_sessions) == 1

    @pytest.mark.asyncio
    async def test_list_sessions_ordered_by_updated_at_desc(self, service) -> None:
        s1 = await service.create_session("alice", "First", "local")
        s2 = await service.create_session("alice", "Second", "local")
        # Add a message to s1 to update its updated_at
        await service.add_message(s1.id, "user", "hello")

        sessions = await service.list_sessions("alice")
        # s1 should be first (most recently updated)
        assert sessions[0].id == s1.id

    @pytest.mark.asyncio
    async def test_archive_session(self, service) -> None:
        session = await service.create_session("alice", "To Archive", "local")
        await service.add_message(session.id, "user", "hello")
        await service.archive_session(session.id)

        with pytest.raises(ValueError):
            await service.get_session(session.id)

        messages = await service.get_messages(session.id)
        assert len(messages) == 0


class TestMessagePersistence:
    """Tests for chat message add and retrieval."""

    @pytest.mark.asyncio
    async def test_add_and_get_messages(self, service) -> None:
        session = await service.create_session("alice", "Chat", "local")
        msg1 = await service.add_message(session.id, "user", "Hello")
        msg2 = await service.add_message(session.id, "assistant", "Hi there")

        assert isinstance(msg1, ChatMessageRecord)
        assert msg1.role == "user"
        assert msg1.content == "Hello"

        messages = await service.get_messages(session.id)
        assert len(messages) == 2
        assert messages[0].role == "user"
        assert messages[1].role == "assistant"

    @pytest.mark.asyncio
    async def test_messages_ordered_by_created_at_asc(self, service) -> None:
        session = await service.create_session("alice", "Chat", "local")
        await service.add_message(session.id, "user", "First")
        await service.add_message(session.id, "assistant", "Second")
        await service.add_message(session.id, "user", "Third")

        messages = await service.get_messages(session.id)
        assert [m.content for m in messages] == ["First", "Second", "Third"]

    @pytest.mark.asyncio
    async def test_add_message_with_tool_calls(self, service) -> None:
        session = await service.create_session("alice", "Chat", "local")
        tool_calls_data = {"name": "set_source", "arguments": {"type": "csv"}}
        msg = await service.add_message(
            session.id, "assistant", "Setting source",
            tool_calls=tool_calls_data,
        )
        assert msg.tool_calls is not None

    @pytest.mark.asyncio
    async def test_add_message_updates_session_updated_at(self, service) -> None:
        session = await service.create_session("alice", "Chat", "local")
        original_updated = session.updated_at
        await service.add_message(session.id, "user", "hello")
        refreshed = await service.get_session(session.id)
        assert refreshed.updated_at >= original_updated


class TestCompositionStateVersioning:
    """Tests for immutable state snapshots with monotonic versioning."""

    @pytest.mark.asyncio
    async def test_first_state_version_is_1(self, service) -> None:
        session = await service.create_session("alice", "Pipeline", "local")
        state_data = CompositionStateData(is_valid=False)
        state = await service.save_composition_state(session.id, state_data)
        assert isinstance(state, CompositionStateRecord)
        assert state.version == 1
        # New states (not reverts) have no lineage (D2/D7)
        assert state.derived_from_state_id is None

    @pytest.mark.asyncio
    async def test_version_increments_monotonically(self, service) -> None:
        session = await service.create_session("alice", "Pipeline", "local")
        s1 = await service.save_composition_state(
            session.id, CompositionStateData(is_valid=False),
        )
        s2 = await service.save_composition_state(
            session.id, CompositionStateData(is_valid=True),
        )
        assert s1.version == 1
        assert s2.version == 2

    @pytest.mark.asyncio
    async def test_get_current_state_returns_highest_version(
        self, service,
    ) -> None:
        session = await service.create_session("alice", "Pipeline", "local")
        await service.save_composition_state(
            session.id,
            CompositionStateData(
                source={"type": "csv", "path": "old.csv"}, is_valid=False,
            ),
        )
        await service.save_composition_state(
            session.id,
            CompositionStateData(
                source={"type": "csv", "path": "new.csv"}, is_valid=True,
            ),
        )
        current = await service.get_current_state(session.id)
        assert current is not None
        assert current.version == 2
        assert current.is_valid is True

    @pytest.mark.asyncio
    async def test_get_current_state_returns_none_when_empty(
        self, service,
    ) -> None:
        session = await service.create_session("alice", "Empty", "local")
        current = await service.get_current_state(session.id)
        assert current is None

    @pytest.mark.asyncio
    async def test_get_state_versions_returns_all_ascending(
        self, service,
    ) -> None:
        session = await service.create_session("alice", "Pipeline", "local")
        await service.save_composition_state(
            session.id, CompositionStateData(is_valid=False),
        )
        await service.save_composition_state(
            session.id, CompositionStateData(is_valid=False),
        )
        await service.save_composition_state(
            session.id, CompositionStateData(is_valid=True),
        )
        versions = await service.get_state_versions(session.id)
        assert len(versions) == 3
        assert [v.version for v in versions] == [1, 2, 3]

    @pytest.mark.asyncio
    async def test_state_preserves_pipeline_data(self, service) -> None:
        session = await service.create_session("alice", "Pipeline", "local")
        state_data = CompositionStateData(
            source={"type": "csv", "path": "/data/input.csv"},
            nodes=[{"name": "classify", "type": "transform"}],
            edges=[{"from": "source", "to": "classify"}],
            outputs=[{"name": "results", "type": "csv_sink"}],
            metadata_={"pipeline_name": "Test Pipeline"},
            is_valid=True,
            validation_errors=None,
        )
        state = await service.save_composition_state(session.id, state_data)
        assert state.is_valid is True


class TestOneActiveRunEnforcement:
    """Tests for B6 -- one active run per session."""

    @pytest.mark.asyncio
    async def test_second_pending_run_raises(self, service) -> None:
        session = await service.create_session("alice", "Pipeline", "local")
        state = await service.save_composition_state(
            session.id, CompositionStateData(is_valid=True),
        )
        # First run should succeed
        await service.create_run(session.id, state.id)
        # Second run should fail
        with pytest.raises(RunAlreadyActiveError):
            await service.create_run(session.id, state.id)

    @pytest.mark.asyncio
    async def test_create_run_returns_run_record(self, service) -> None:
        session = await service.create_session("alice", "Pipeline", "local")
        state = await service.save_composition_state(
            session.id, CompositionStateData(is_valid=True),
        )
        run = await service.create_run(session.id, state.id)
        assert isinstance(run, RunRecord)
        assert run.status == "pending"
        assert run.session_id == session.id
        assert run.state_id == state.id
        assert run.pipeline_yaml is None

    @pytest.mark.asyncio
    async def test_create_run_with_pipeline_yaml(self, service) -> None:
        session = await service.create_session("alice", "Pipeline", "local")
        state = await service.save_composition_state(
            session.id, CompositionStateData(is_valid=True),
        )
        run = await service.create_run(
            session.id, state.id, pipeline_yaml="source:\n  type: csv",
        )
        assert run.pipeline_yaml == "source:\n  type: csv"

    @pytest.mark.asyncio
    async def test_completed_run_allows_new_run(self, service) -> None:
        session = await service.create_session("alice", "Pipeline", "local")
        state = await service.save_composition_state(
            session.id, CompositionStateData(is_valid=True),
        )
        run = await service.create_run(session.id, state.id)
        # Transition through legal path: pending → running → completed
        await service.update_run_status(run.id, "running")
        await service.update_run_status(run.id, "completed")
        # New run should succeed
        run2 = await service.create_run(session.id, state.id)
        assert run2.status == "pending"

    @pytest.mark.asyncio
    async def test_failed_run_allows_new_run(self, service) -> None:
        session = await service.create_session("alice", "Pipeline", "local")
        state = await service.save_composition_state(
            session.id, CompositionStateData(is_valid=True),
        )
        run = await service.create_run(session.id, state.id)
        # Transition through legal path: pending → running → failed
        await service.update_run_status(run.id, "running")
        await service.update_run_status(run.id, "failed")
        run2 = await service.create_run(session.id, state.id)
        assert run2.status == "pending"

    @pytest.mark.asyncio
    async def test_running_run_blocks_new_run(self, service) -> None:
        session = await service.create_session("alice", "Pipeline", "local")
        state = await service.save_composition_state(
            session.id, CompositionStateData(is_valid=True),
        )
        run = await service.create_run(session.id, state.id)
        await service.update_run_status(run.id, "running")
        with pytest.raises(RunAlreadyActiveError):
            await service.create_run(session.id, state.id)


class TestGetState:
    """Tests for get_state -- fetch a specific CompositionStateRecord by UUID."""

    @pytest.mark.asyncio
    async def test_get_state_by_id(self, service) -> None:
        session = await service.create_session("alice", "Pipeline", "local")
        saved = await service.save_composition_state(
            session.id,
            CompositionStateData(
                source={"type": "csv"}, is_valid=True,
            ),
        )
        fetched = await service.get_state(saved.id)
        assert fetched.id == saved.id
        assert fetched.version == saved.version

    @pytest.mark.asyncio
    async def test_get_state_not_found_raises(self, service) -> None:
        with pytest.raises(ValueError, match="not found"):
            await service.get_state(uuid.uuid4())


class TestSetActiveState:
    """Tests for set_active_state -- revert by copying a prior version."""

    @pytest.mark.asyncio
    async def test_revert_creates_new_version(self, service) -> None:
        session = await service.create_session("alice", "Pipeline", "local")
        v1 = await service.save_composition_state(
            session.id,
            CompositionStateData(source={"type": "csv"}, is_valid=True),
        )
        v2 = await service.save_composition_state(
            session.id,
            CompositionStateData(source={"type": "api"}, is_valid=True),
        )
        # Revert to v1 -- should create v3 as a copy of v1
        reverted = await service.set_active_state(session.id, v1.id)
        assert reverted.version == 3
        # Content should match v1, not v2
        assert reverted.source == v1.source
        # Lineage: reverted state records where it came from (D6)
        assert reverted.derived_from_state_id == v1.id

    @pytest.mark.asyncio
    async def test_revert_preserves_history(self, service) -> None:
        session = await service.create_session("alice", "Pipeline", "local")
        await service.save_composition_state(
            session.id, CompositionStateData(is_valid=False),
        )
        v2 = await service.save_composition_state(
            session.id, CompositionStateData(is_valid=True),
        )
        await service.set_active_state(session.id, v2.id)
        versions = await service.get_state_versions(session.id)
        # All three versions should exist (v1, v2, v3)
        assert len(versions) == 3
        assert [v.version for v in versions] == [1, 2, 3]

    @pytest.mark.asyncio
    async def test_revert_state_not_found_raises(self, service) -> None:
        session = await service.create_session("alice", "Pipeline", "local")
        with pytest.raises(ValueError, match="not found"):
            await service.set_active_state(session.id, uuid.uuid4())

    @pytest.mark.asyncio
    async def test_revert_state_wrong_session_raises(self, service) -> None:
        s1 = await service.create_session("alice", "Session 1", "local")
        s2 = await service.create_session("alice", "Session 2", "local")
        state = await service.save_composition_state(
            s1.id, CompositionStateData(is_valid=True),
        )
        with pytest.raises(ValueError, match="does not belong"):
            await service.set_active_state(s2.id, state.id)


class TestGetRun:
    """Tests for get_run -- fetch a RunRecord by UUID."""

    @pytest.mark.asyncio
    async def test_get_run_returns_record(self, service) -> None:
        session = await service.create_session("alice", "Pipeline", "local")
        state = await service.save_composition_state(
            session.id, CompositionStateData(is_valid=True),
        )
        created = await service.create_run(session.id, state.id)
        fetched = await service.get_run(created.id)
        assert isinstance(fetched, RunRecord)
        assert fetched.id == created.id
        assert fetched.status == "pending"

    @pytest.mark.asyncio
    async def test_get_run_not_found_raises(self, service) -> None:
        with pytest.raises(ValueError, match="not found"):
            await service.get_run(uuid.uuid4())


class TestGetActiveRun:
    """Tests for get_active_run -- pending/running run for a session."""

    @pytest.mark.asyncio
    async def test_returns_active_run(self, service) -> None:
        session = await service.create_session("alice", "Pipeline", "local")
        state = await service.save_composition_state(
            session.id, CompositionStateData(is_valid=True),
        )
        run = await service.create_run(session.id, state.id)
        active = await service.get_active_run(session.id)
        assert active is not None
        assert active.id == run.id

    @pytest.mark.asyncio
    async def test_returns_none_when_no_active_run(self, service) -> None:
        session = await service.create_session("alice", "Pipeline", "local")
        active = await service.get_active_run(session.id)
        assert active is None

    @pytest.mark.asyncio
    async def test_returns_none_after_completion(self, service) -> None:
        session = await service.create_session("alice", "Pipeline", "local")
        state = await service.save_composition_state(
            session.id, CompositionStateData(is_valid=True),
        )
        run = await service.create_run(session.id, state.id)
        await service.update_run_status(run.id, "running")
        await service.update_run_status(run.id, "completed")
        active = await service.get_active_run(session.id)
        assert active is None


class TestUpdateRunStatusExpanded:
    """Tests for expanded update_run_status signature (R6)."""

    @pytest.mark.asyncio
    async def test_update_with_error(self, service) -> None:
        session = await service.create_session("alice", "Pipeline", "local")
        state = await service.save_composition_state(
            session.id, CompositionStateData(is_valid=True),
        )
        run = await service.create_run(session.id, state.id)
        await service.update_run_status(run.id, "running")
        await service.update_run_status(
            run.id, "failed", error="Source file not found",
        )
        fetched = await service.get_run(run.id)
        assert fetched.status == "failed"
        assert fetched.error == "Source file not found"
        assert fetched.finished_at is not None

    @pytest.mark.asyncio
    async def test_update_with_landscape_run_id(self, service) -> None:
        session = await service.create_session("alice", "Pipeline", "local")
        state = await service.save_composition_state(
            session.id, CompositionStateData(is_valid=True),
        )
        run = await service.create_run(session.id, state.id)
        await service.update_run_status(run.id, "running")
        await service.update_run_status(
            run.id, "completed",
            landscape_run_id="lscp-abc-123",
            rows_processed=100,
            rows_failed=3,
        )
        fetched = await service.get_run(run.id)
        assert fetched.landscape_run_id == "lscp-abc-123"
        assert fetched.rows_processed == 100
        assert fetched.rows_failed == 3

    @pytest.mark.asyncio
    async def test_update_not_found_raises(self, service) -> None:
        with pytest.raises(ValueError, match="not found"):
            await service.update_run_status(uuid.uuid4(), "completed")


class TestRunTransitionEnforcement:
    """Tests for D3 -- LEGAL_RUN_TRANSITIONS enforcement."""

    @pytest.mark.asyncio
    async def test_legal_transition_pending_to_running(self, service) -> None:
        session = await service.create_session("alice", "Pipeline", "local")
        state = await service.save_composition_state(
            session.id, CompositionStateData(is_valid=True),
        )
        run = await service.create_run(session.id, state.id)
        await service.update_run_status(run.id, "running")
        fetched = await service.get_run(run.id)
        assert fetched.status == "running"

    @pytest.mark.asyncio
    async def test_legal_transition_pending_to_cancelled(self, service) -> None:
        session = await service.create_session("alice", "Pipeline", "local")
        state = await service.save_composition_state(
            session.id, CompositionStateData(is_valid=True),
        )
        run = await service.create_run(session.id, state.id)
        await service.update_run_status(run.id, "cancelled")
        fetched = await service.get_run(run.id)
        assert fetched.status == "cancelled"
        assert fetched.finished_at is not None

    @pytest.mark.asyncio
    async def test_illegal_transition_pending_to_completed_raises(
        self, service,
    ) -> None:
        session = await service.create_session("alice", "Pipeline", "local")
        state = await service.save_composition_state(
            session.id, CompositionStateData(is_valid=True),
        )
        run = await service.create_run(session.id, state.id)
        with pytest.raises(ValueError, match="Illegal.*transition"):
            await service.update_run_status(run.id, "completed")

    @pytest.mark.asyncio
    async def test_illegal_transition_completed_to_running_raises(
        self, service,
    ) -> None:
        session = await service.create_session("alice", "Pipeline", "local")
        state = await service.save_composition_state(
            session.id, CompositionStateData(is_valid=True),
        )
        run = await service.create_run(session.id, state.id)
        await service.update_run_status(run.id, "running")
        await service.update_run_status(run.id, "completed")
        with pytest.raises(ValueError, match="Illegal.*transition"):
            await service.update_run_status(run.id, "running")


class TestLandscapeRunIdWriteOnce:
    """Tests for D4 -- landscape_run_id is write-once."""

    @pytest.mark.asyncio
    async def test_set_landscape_run_id(self, service) -> None:
        session = await service.create_session("alice", "Pipeline", "local")
        state = await service.save_composition_state(
            session.id, CompositionStateData(is_valid=True),
        )
        run = await service.create_run(session.id, state.id)
        await service.update_run_status(
            run.id, "running", landscape_run_id="lscp-001",
        )
        fetched = await service.get_run(run.id)
        assert fetched.landscape_run_id == "lscp-001"

    @pytest.mark.asyncio
    async def test_overwrite_landscape_run_id_raises(self, service) -> None:
        session = await service.create_session("alice", "Pipeline", "local")
        state = await service.save_composition_state(
            session.id, CompositionStateData(is_valid=True),
        )
        run = await service.create_run(session.id, state.id)
        await service.update_run_status(
            run.id, "running", landscape_run_id="lscp-001",
        )
        with pytest.raises(ValueError, match="landscape_run_id.*already set"):
            await service.update_run_status(
                run.id, "completed", landscape_run_id="lscp-002",
            )

    @pytest.mark.asyncio
    async def test_none_landscape_run_id_does_not_overwrite(
        self, service,
    ) -> None:
        session = await service.create_session("alice", "Pipeline", "local")
        state = await service.save_composition_state(
            session.id, CompositionStateData(is_valid=True),
        )
        run = await service.create_run(session.id, state.id)
        await service.update_run_status(
            run.id, "running", landscape_run_id="lscp-001",
        )
        # Passing None (default) should not trigger the write-once guard
        await service.update_run_status(run.id, "completed")
        fetched = await service.get_run(run.id)
        assert fetched.landscape_run_id == "lscp-001"


class TestCancelOrphanedRuns:
    """Tests for D5 -- cancel_orphaned_runs."""

    @pytest.mark.asyncio
    async def test_cancels_stale_running_run(self, service) -> None:
        session = await service.create_session("alice", "Pipeline", "local")
        state = await service.save_composition_state(
            session.id, CompositionStateData(is_valid=True),
        )
        run = await service.create_run(session.id, state.id)
        await service.update_run_status(run.id, "running")
        # Cancel with max_age_seconds=0 so ANY running run is considered stale
        cancelled = await service.cancel_orphaned_runs(
            session.id, max_age_seconds=0,
        )
        assert len(cancelled) == 1
        assert cancelled[0].id == run.id
        assert cancelled[0].status == "cancelled"

    @pytest.mark.asyncio
    async def test_does_not_cancel_recent_running_run(self, service) -> None:
        session = await service.create_session("alice", "Pipeline", "local")
        state = await service.save_composition_state(
            session.id, CompositionStateData(is_valid=True),
        )
        run = await service.create_run(session.id, state.id)
        await service.update_run_status(run.id, "running")
        # max_age_seconds=3600 -- run was just created, so not stale
        cancelled = await service.cancel_orphaned_runs(
            session.id, max_age_seconds=3600,
        )
        assert len(cancelled) == 0

    @pytest.mark.asyncio
    async def test_does_not_cancel_completed_runs(self, service) -> None:
        session = await service.create_session("alice", "Pipeline", "local")
        state = await service.save_composition_state(
            session.id, CompositionStateData(is_valid=True),
        )
        run = await service.create_run(session.id, state.id)
        await service.update_run_status(run.id, "running")
        await service.update_run_status(run.id, "completed")
        cancelled = await service.cancel_orphaned_runs(
            session.id, max_age_seconds=0,
        )
        assert len(cancelled) == 0

    @pytest.mark.asyncio
    async def test_cancel_unblocks_session_for_new_run(self, service) -> None:
        session = await service.create_session("alice", "Pipeline", "local")
        state = await service.save_composition_state(
            session.id, CompositionStateData(is_valid=True),
        )
        run = await service.create_run(session.id, state.id)
        await service.update_run_status(run.id, "running")
        await service.cancel_orphaned_runs(session.id, max_age_seconds=0)
        # Session should now accept a new run
        run2 = await service.create_run(session.id, state.id)
        assert run2.status == "pending"
```

- [ ] **Step 2: Run tests -- verify they fail**

```bash
.venv/bin/python -m pytest tests/unit/web/sessions/test_service.py -v
```

Expected: `ModuleNotFoundError: No module named 'elspeth.web.sessions.service'`

- [ ] **Step 3: Implement SessionServiceImpl**

```python
# src/elspeth/web/sessions/service.py
"""SessionService implementation -- CRUD, state versioning, active run enforcement.

Uses SQLAlchemy Core with a synchronous engine. Each method executes SQL
within a single transaction.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from sqlalchemy import Engine, delete, desc, func, insert, select, update

from elspeth.web.sessions.models import (
    chat_messages_table,
    composition_states_table,
    run_events_table,
    runs_table,
    sessions_table,
)
from elspeth.web.sessions.protocol import (
    LEGAL_RUN_TRANSITIONS,
    ChatMessageRecord,
    CompositionStateData,
    CompositionStateRecord,
    RunAlreadyActiveError,
    RunRecord,
    SessionRecord,
)


class SessionServiceImpl:
    """Concrete session service backed by SQLAlchemy Core."""

    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    def _now(self) -> datetime:
        return datetime.now(timezone.utc)

    async def create_session(
        self, user_id: str, title: str, auth_provider_type: str,
    ) -> SessionRecord:
        """Create a new session and return its record."""
        session_id = uuid.uuid4()
        now = self._now()

        with self._engine.begin() as conn:
            conn.execute(
                insert(sessions_table).values(
                    id=str(session_id),
                    user_id=user_id,
                    auth_provider_type=auth_provider_type,
                    title=title,
                    created_at=now,
                    updated_at=now,
                )
            )

        return SessionRecord(
            id=session_id,
            user_id=user_id,
            auth_provider_type=auth_provider_type,
            title=title,
            created_at=now,
            updated_at=now,
        )

    async def get_session(self, session_id: UUID) -> SessionRecord:
        """Fetch a session by ID. Raises ValueError if not found."""
        with self._engine.begin() as conn:
            row = conn.execute(
                select(sessions_table).where(
                    sessions_table.c.id == str(session_id)
                )
            ).fetchone()

        if row is None:
            raise ValueError(f"Session not found: {session_id}")

        return SessionRecord(
            id=UUID(row.id),
            user_id=row.user_id,
            auth_provider_type=row.auth_provider_type,
            title=row.title,
            created_at=row.created_at,
            updated_at=row.updated_at,
        )

    async def list_sessions(self, user_id: str) -> list[SessionRecord]:
        """List sessions for a user, ordered by updated_at descending."""
        with self._engine.begin() as conn:
            rows = conn.execute(
                select(sessions_table)
                .where(sessions_table.c.user_id == user_id)
                .order_by(desc(sessions_table.c.updated_at))
            ).fetchall()

        return [
            SessionRecord(
                id=UUID(row.id),
                user_id=row.user_id,
                auth_provider_type=row.auth_provider_type,
                title=row.title,
                created_at=row.created_at,
                updated_at=row.updated_at,
            )
            for row in rows
        ]

    async def archive_session(self, session_id: UUID) -> None:
        """Delete a session and cascade to all related records."""
        sid = str(session_id)
        with self._engine.begin() as conn:
            # Delete in dependency order (children first for non-CASCADE DBs)
            # Get run IDs for this session to delete run_events
            run_ids = [
                r.id for r in conn.execute(
                    select(runs_table.c.id).where(
                        runs_table.c.session_id == sid
                    )
                ).fetchall()
            ]
            if run_ids:
                conn.execute(
                    delete(run_events_table).where(
                        run_events_table.c.run_id.in_(run_ids)
                    )
                )
            conn.execute(
                delete(runs_table).where(runs_table.c.session_id == sid)
            )
            conn.execute(
                delete(composition_states_table).where(
                    composition_states_table.c.session_id == sid
                )
            )
            conn.execute(
                delete(chat_messages_table).where(
                    chat_messages_table.c.session_id == sid
                )
            )
            conn.execute(
                delete(sessions_table).where(sessions_table.c.id == sid)
            )

    async def add_message(
        self,
        session_id: UUID,
        role: str,
        content: str,
        tool_calls: dict[str, Any] | None = None,
    ) -> ChatMessageRecord:
        """Add a chat message and update the session's updated_at."""
        msg_id = uuid.uuid4()
        now = self._now()
        sid = str(session_id)

        with self._engine.begin() as conn:
            conn.execute(
                insert(chat_messages_table).values(
                    id=str(msg_id),
                    session_id=sid,
                    role=role,
                    content=content,
                    tool_calls=tool_calls,
                    created_at=now,
                )
            )
            conn.execute(
                update(sessions_table)
                .where(sessions_table.c.id == sid)
                .values(updated_at=now)
            )

        return ChatMessageRecord(
            id=msg_id,
            session_id=session_id,
            role=role,
            content=content,
            tool_calls=tool_calls,
            created_at=now,
        )

    async def get_messages(
        self, session_id: UUID,
    ) -> list[ChatMessageRecord]:
        """Get all messages for a session, ordered by created_at ascending."""
        with self._engine.begin() as conn:
            rows = conn.execute(
                select(chat_messages_table)
                .where(
                    chat_messages_table.c.session_id == str(session_id)
                )
                .order_by(chat_messages_table.c.created_at)
            ).fetchall()

        return [
            ChatMessageRecord(
                id=UUID(row.id),
                session_id=UUID(row.session_id),
                role=row.role,
                content=row.content,
                tool_calls=row.tool_calls,
                created_at=row.created_at,
            )
            for row in rows
        ]

    async def save_composition_state(
        self, session_id: UUID, state: CompositionStateData,
    ) -> CompositionStateRecord:
        """Save a new immutable composition state snapshot.

        Version is max(existing versions for session) + 1, starting at 1.
        """
        state_id = uuid.uuid4()
        now = self._now()
        sid = str(session_id)

        with self._engine.begin() as conn:
            # Get the current max version for this session
            result = conn.execute(
                select(func.max(composition_states_table.c.version)).where(
                    composition_states_table.c.session_id == sid
                )
            ).scalar()
            version = (result or 0) + 1

            # Unfreeze data for JSON serialization (MappingProxyType is not
            # JSON-serializable). Convert back to plain dicts/lists.
            def _to_json(val: Any) -> Any:
                if val is None:
                    return None
                if isinstance(val, dict):
                    return dict(val)
                if isinstance(val, (list, tuple)):
                    return list(val)
                return val

            # Seam contract A: wrap JSON columns with _version envelope
            # for schema evolution. Version 1 is the initial format.
            def _enveloped(val: Any) -> Any:
                raw = _to_json(val)
                if raw is None:
                    return None
                return {"_version": 1, "data": raw}

            conn.execute(
                insert(composition_states_table).values(
                    id=str(state_id),
                    session_id=sid,
                    version=version,
                    source=_enveloped(state.source),
                    nodes=_enveloped(state.nodes),
                    edges=_enveloped(state.edges),
                    outputs=_enveloped(state.outputs),
                    metadata_=_enveloped(state.metadata_),
                    is_valid=state.is_valid,
                    validation_errors=_to_json(state.validation_errors),
                    derived_from_state_id=None,
                    created_at=now,
                )
            )

        return CompositionStateRecord(
            id=state_id,
            session_id=session_id,
            version=version,
            source=state.source,
            nodes=state.nodes,
            edges=state.edges,
            outputs=state.outputs,
            metadata_=state.metadata_,
            is_valid=state.is_valid,
            validation_errors=state.validation_errors,
            created_at=now,
            derived_from_state_id=None,
        )

    async def get_current_state(
        self, session_id: UUID,
    ) -> CompositionStateRecord | None:
        """Return the highest-version state for a session, or None."""
        with self._engine.begin() as conn:
            row = conn.execute(
                select(composition_states_table)
                .where(
                    composition_states_table.c.session_id == str(session_id)
                )
                .order_by(desc(composition_states_table.c.version))
                .limit(1)
            ).fetchone()

        if row is None:
            return None

        return self._row_to_state_record(row)

    async def get_state_versions(
        self, session_id: UUID,
    ) -> list[CompositionStateRecord]:
        """Return all state versions for a session, ascending order."""
        with self._engine.begin() as conn:
            rows = conn.execute(
                select(composition_states_table)
                .where(
                    composition_states_table.c.session_id == str(session_id)
                )
                .order_by(composition_states_table.c.version)
            ).fetchall()

        return [self._row_to_state_record(row) for row in rows]

    @staticmethod
    def _unwrap_envelope(val: Any) -> Any:
        """Unwrap _version envelope from a JSON column value.

        Seam contract A: JSON columns are stored with {"_version": 1, "data": ...}.
        Raises ValueError on unknown versions. Returns None for NULL columns.
        """
        if val is None:
            return None
        if isinstance(val, dict) and "_version" in val:
            if val["_version"] != 1:
                raise ValueError(
                    f"Unknown composition state envelope version: {val['_version']}"
                )
            return val["data"]
        # Legacy data without envelope -- pass through as-is
        return val

    def _row_to_state_record(self, row: Any) -> CompositionStateRecord:
        """Convert a SQLAlchemy row to a CompositionStateRecord.

        Seam contract A: metadata_ maps DB column metadata_ back to the
        dataclass field. JSON columns are unwrapped from their _version envelope.
        When Sub-4 provides CompositionState.from_dict(), this method becomes
        the swap point for domain object reconstruction.
        """
        # TODO(sub-4): Replace raw dict fields with CompositionState.from_dict()
        # call once Sub-4 defines it. The swap should be a single-line change.
        return CompositionStateRecord(
            id=UUID(row.id),
            session_id=UUID(row.session_id),
            version=row.version,
            source=self._unwrap_envelope(row.source),
            nodes=self._unwrap_envelope(row.nodes),
            edges=self._unwrap_envelope(row.edges),
            outputs=self._unwrap_envelope(row.outputs),
            metadata_=self._unwrap_envelope(row.metadata_),
            is_valid=row.is_valid,
            validation_errors=row.validation_errors,
            created_at=row.created_at,
            derived_from_state_id=(
                UUID(row.derived_from_state_id)
                if row.derived_from_state_id is not None
                else None
            ),
        )

    async def create_run(
        self,
        session_id: UUID,
        state_id: UUID,
        pipeline_yaml: str | None = None,
    ) -> RunRecord:
        """Create a new pending run, enforcing one active run per session (B6).

        The check-and-set runs within the same database transaction.
        Raises RunAlreadyActiveError if a pending or running run exists.
        If pipeline_yaml is provided, stores the generated YAML at creation time.
        """
        run_id = uuid.uuid4()
        now = self._now()
        sid = str(session_id)

        with self._engine.begin() as conn:
            # Check for existing active runs
            active = conn.execute(
                select(runs_table.c.id).where(
                    runs_table.c.session_id == sid,
                    runs_table.c.status.in_(["pending", "running"]),
                )
            ).fetchone()

            if active is not None:
                raise RunAlreadyActiveError(sid)

            conn.execute(
                insert(runs_table).values(
                    id=str(run_id),
                    session_id=sid,
                    state_id=str(state_id),
                    status="pending",
                    started_at=now,
                    rows_processed=0,
                    rows_failed=0,
                    pipeline_yaml=pipeline_yaml,
                )
            )

        return RunRecord(
            id=run_id,
            session_id=session_id,
            state_id=state_id,
            status="pending",
            started_at=now,
            finished_at=None,
            rows_processed=0,
            rows_failed=0,
            error=None,
            landscape_run_id=None,
            pipeline_yaml=pipeline_yaml,
        )

    async def get_run(self, run_id: UUID) -> RunRecord:
        """Fetch a run by ID. Raises ValueError if not found."""
        with self._engine.begin() as conn:
            row = conn.execute(
                select(runs_table).where(
                    runs_table.c.id == str(run_id)
                )
            ).fetchone()

        if row is None:
            raise ValueError(f"Run not found: {run_id}")

        return self._row_to_run_record(row)

    async def update_run_status(
        self,
        run_id: UUID,
        status: str,
        error: str | None = None,
        landscape_run_id: str | None = None,
        rows_processed: int | None = None,
        rows_failed: int | None = None,
    ) -> None:
        """Update a run's status and optional fields.

        Enforces LEGAL_RUN_TRANSITIONS (D3). Enforces landscape_run_id
        write-once semantics (D4). Sets finished_at for terminal states
        (completed, failed, cancelled). Optional parameters only update
        the column when not None. Raises ValueError if run not found or
        transition is illegal.
        """
        now = self._now()
        rid = str(run_id)

        with self._engine.begin() as conn:
            # Read current state for transition + write-once validation
            current = conn.execute(
                select(
                    runs_table.c.status,
                    runs_table.c.landscape_run_id,
                ).where(runs_table.c.id == rid)
            ).fetchone()

            if current is None:
                raise ValueError(f"Run not found: {run_id}")

            # D3: Enforce legal transitions
            current_status = current.status
            allowed = LEGAL_RUN_TRANSITIONS.get(current_status, frozenset())
            if status not in allowed:
                raise ValueError(
                    f"Illegal run transition: {current_status!r} → {status!r}. "
                    f"Allowed: {sorted(allowed)}"
                )

            # D4: landscape_run_id is write-once
            if (
                landscape_run_id is not None
                and current.landscape_run_id is not None
            ):
                raise ValueError(
                    f"landscape_run_id already set to "
                    f"{current.landscape_run_id!r}; cannot overwrite"
                )

            values: dict[str, Any] = {"status": status}
            if status in ("completed", "failed", "cancelled"):
                values["finished_at"] = now
            if error is not None:
                values["error"] = error
            if landscape_run_id is not None:
                values["landscape_run_id"] = landscape_run_id
            if rows_processed is not None:
                values["rows_processed"] = rows_processed
            if rows_failed is not None:
                values["rows_failed"] = rows_failed

            conn.execute(
                update(runs_table)
                .where(runs_table.c.id == rid)
                .values(**values)
            )

    async def get_active_run(
        self, session_id: UUID,
    ) -> RunRecord | None:
        """Return the pending/running run for a session, or None."""
        with self._engine.begin() as conn:
            row = conn.execute(
                select(runs_table).where(
                    runs_table.c.session_id == str(session_id),
                    runs_table.c.status.in_(["pending", "running"]),
                )
            ).fetchone()

        if row is None:
            return None

        return self._row_to_run_record(row)

    async def get_state(self, state_id: UUID) -> CompositionStateRecord:
        """Fetch a composition state by its primary key. Raises ValueError if not found."""
        with self._engine.begin() as conn:
            row = conn.execute(
                select(composition_states_table).where(
                    composition_states_table.c.id == str(state_id)
                )
            ).fetchone()

        if row is None:
            raise ValueError(f"State not found: {state_id}")

        return self._row_to_state_record(row)

    async def set_active_state(
        self, session_id: UUID, state_id: UUID,
    ) -> CompositionStateRecord:
        """Revert to a prior state by copying it as a new version.

        Creates a new version record that is a copy of the specified prior
        version (looked up by state_id). The new record gets
        version = max(existing) + 1. Raises ValueError if state_id not
        found or does not belong to the session.
        """
        sid = str(session_id)

        with self._engine.begin() as conn:
            # Look up the prior state
            prior_row = conn.execute(
                select(composition_states_table).where(
                    composition_states_table.c.id == str(state_id)
                )
            ).fetchone()

            if prior_row is None:
                raise ValueError(f"State not found: {state_id}")
            if prior_row.session_id != sid:
                raise ValueError(
                    f"State {state_id} does not belong to session {session_id}"
                )

            # Get next version number
            max_version = conn.execute(
                select(func.max(composition_states_table.c.version)).where(
                    composition_states_table.c.session_id == sid
                )
            ).scalar()
            new_version = (max_version or 0) + 1

            new_state_id = uuid.uuid4()
            now = self._now()

            conn.execute(
                insert(composition_states_table).values(
                    id=str(new_state_id),
                    session_id=sid,
                    version=new_version,
                    source=prior_row.source,
                    nodes=prior_row.nodes,
                    edges=prior_row.edges,
                    outputs=prior_row.outputs,
                    metadata_=prior_row.metadata_,
                    is_valid=prior_row.is_valid,
                    validation_errors=prior_row.validation_errors,
                    derived_from_state_id=str(state_id),
                    created_at=now,
                )
            )

        return CompositionStateRecord(
            id=new_state_id,
            session_id=session_id,
            version=new_version,
            source=self._unwrap_envelope(prior_row.source),
            nodes=self._unwrap_envelope(prior_row.nodes),
            edges=self._unwrap_envelope(prior_row.edges),
            outputs=self._unwrap_envelope(prior_row.outputs),
            metadata_=self._unwrap_envelope(prior_row.metadata_),
            is_valid=prior_row.is_valid,
            validation_errors=prior_row.validation_errors,
            created_at=now,
            derived_from_state_id=state_id,
        )

    async def cancel_orphaned_runs(
        self,
        session_id: UUID,
        max_age_seconds: int = 3600,
    ) -> list[RunRecord]:
        """Force-cancel runs stuck in 'running' status beyond max_age_seconds.

        Returns the list of cancelled RunRecords. Called by the execution
        service on startup and periodically to prevent orphaned runs from
        permanently blocking sessions (D5).
        """
        sid = str(session_id)
        now = self._now()
        cutoff = datetime.fromtimestamp(
            now.timestamp() - max_age_seconds, tz=timezone.utc,
        )

        cancelled: list[RunRecord] = []

        with self._engine.begin() as conn:
            # Find running runs older than the cutoff
            stale_rows = conn.execute(
                select(runs_table).where(
                    runs_table.c.session_id == sid,
                    runs_table.c.status == "running",
                    runs_table.c.started_at <= cutoff,
                )
            ).fetchall()

            for row in stale_rows:
                conn.execute(
                    update(runs_table)
                    .where(runs_table.c.id == row.id)
                    .values(status="cancelled", finished_at=now)
                )
                cancelled.append(RunRecord(
                    id=UUID(row.id),
                    session_id=UUID(row.session_id),
                    state_id=UUID(row.state_id),
                    status="cancelled",
                    started_at=row.started_at,
                    finished_at=now,
                    rows_processed=row.rows_processed,
                    rows_failed=row.rows_failed,
                    error=row.error,
                    landscape_run_id=row.landscape_run_id,
                    pipeline_yaml=row.pipeline_yaml,
                ))

        return cancelled

    def _row_to_run_record(self, row: Any) -> RunRecord:
        """Convert a SQLAlchemy row to a RunRecord."""
        return RunRecord(
            id=UUID(row.id),
            session_id=UUID(row.session_id),
            state_id=UUID(row.state_id),
            status=row.status,
            started_at=row.started_at,
            finished_at=row.finished_at,
            rows_processed=row.rows_processed,
            rows_failed=row.rows_failed,
            error=row.error,
            landscape_run_id=row.landscape_run_id,
            pipeline_yaml=row.pipeline_yaml,
        )
```

- [ ] **Step 4: Run tests -- verify they pass**

```bash
.venv/bin/python -m pytest tests/unit/web/sessions/test_service.py -v
```

Expected: all 47 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/elspeth/web/sessions/service.py tests/unit/web/sessions/test_service.py
git commit -m "feat(web/sessions): implement SessionServiceImpl with CRUD, versioning, and active run enforcement"
```

---

## Self-Review Checklist

- [ ] `src/elspeth/web/sessions/service.py` creates `SessionServiceImpl` class with sync SQLAlchemy Core
- [ ] All CRUD operations: `create_session`, `get_session`, `list_sessions`, `archive_session`
- [ ] Message persistence: `add_message`, `get_messages`
- [ ] Composition state versioning: `save_composition_state`, `get_current_state`, `get_state_versions`, `get_state`
- [ ] State revert: `set_active_state` copies prior version as new highest version
- [ ] Run enforcement (B6): `create_run` checks for active runs in same transaction, raises `RunAlreadyActiveError`
- [ ] Run lifecycle: `get_run`, `get_active_run`, `update_run_status` with expanded signature (R6)
- [ ] `update_run_status` enforces `LEGAL_RUN_TRANSITIONS` — illegal transitions raise `ValueError` (D3)
- [ ] `update_run_status` enforces `landscape_run_id` write-once — double-write raises `ValueError` (D4)
- [ ] `update_run_status` sets `finished_at` for terminal states, accepts `error`, `landscape_run_id`, `rows_processed`, `rows_failed`
- [ ] `cancel_orphaned_runs` force-cancels stale running runs and returns cancelled records (D5)
- [ ] `create_session` accepts and stores `auth_provider_type` (D1)
- [ ] `save_composition_state` sets `derived_from_state_id=None` in INSERT (D7)
- [ ] `set_active_state` sets `derived_from_state_id=state_id` — lineage recorded (D6)
- [ ] `set_active_state` unwraps envelopes on returned record (not raw DB JSON)
- [ ] All `CompositionStateRecord` constructions include `derived_from_state_id` (D2)
- [ ] Seam contract A: JSON columns wrapped in `{"_version": 1, "data": ...}` envelope for schema evolution
- [ ] `_row_to_state_record` unwraps envelopes and converts `derived_from_state_id` to UUID; `_row_to_run_record` maps DB rows to records
- [ ] `archive_session` deletes in dependency order (run_events, runs, composition_states, chat_messages, sessions)
- [ ] `add_message` updates session `updated_at` for correct list ordering
- [ ] All 49 tests pass in `tests/unit/web/sessions/test_service.py`
- [ ] Tests cover: CRUD, message ordering, tool_calls, state versioning, run enforcement, revert, expanded status updates, transition enforcement (D3), write-once landscape_run_id (D4), cancel orphaned runs (D5), derived_from_state_id lineage (D2/D6)
- [ ] No imports from layers above L1 (service is L3, imports L0 contracts + L1 models only within web/)
