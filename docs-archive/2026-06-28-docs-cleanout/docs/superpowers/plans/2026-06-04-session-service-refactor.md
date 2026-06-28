# Session Service Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split `src/elspeth/web/sessions/service.py` into focused, test-pinned private modules while preserving `SessionServiceImpl` as the public service facade and preserving every audit, lock, and route-facing behavior.

**Architecture:** Keep `SessionServiceImpl` in `service.py` as the construction/import surface used by the web app, routes, and tests. Extract pure row mappers and interpretation-resolution helpers first, then move cohesive method groups into private mixins that inherit a shared `SessionServiceBase` with the engine, telemetry, logging, async worker bridge, session write locks, and canonical insert helpers. Preserve current exception types and writer preconditions exactly; refactor by moving code, not by redesigning persistence semantics.

**Tech Stack:** Python 3.13, SQLAlchemy Core, FastAPI service layer, pytest, structlog, OpenTelemetry session telemetry, Elspeth trust-tier linting.

---

## Scope And Guardrails

This is a structural refactor of `src/elspeth/web/sessions/service.py` only. It must not change:

- The `SessionServiceImpl(...)` constructor signature.
- The public async methods required by `SessionServiceProtocol`.
- The `SessionServiceImpl._ensure_utc(...)` compatibility hook used by timezone tests.
- The process-wide SQLite session-write-lock behavior.
- The ordering of audit writes before telemetry/logging.
- The interpretation-event exception classes and route error mapping.
- The `composition_states` JSON envelope shape.
- The direct-writer static guard semantics in `tests/unit/web/sessions/test_static_direct_writers.py`.

Current live file structure, from the 2026-06-04 tree:

- `src/elspeth/web/sessions/service.py:154-487`: proposal/state/run row helpers and interpretation row conversion.
- `src/elspeth/web/sessions/service.py:490-1331`: interpretation-resolution error classes and pure helper functions.
- `src/elspeth/web/sessions/service.py:1332-2056`: service constructor, sync bridge, locks, canonical insert helpers, and compose-turn persistence.
- `src/elspeth/web/sessions/service.py:2057-2370`: session CRUD, archive, and composer preferences.
- `src/elspeth/web/sessions/service.py:2370-2600`: composer proposal lifecycle.
- `src/elspeth/web/sessions/service.py:2601-3601`: interpretation-event write/read/resolve methods.
- `src/elspeth/web/sessions/service.py:3602-3872`: chat messages and audit-grade access logging.
- `src/elspeth/web/sessions/service.py:3873-4034`: composition-state save/read mapping.
- `src/elspeth/web/sessions/service.py:4035-4622`: run lifecycle, blob inline resolution audit rows, active-state switching, orphan cancellation, and pruning.
- `src/elspeth/web/sessions/service.py:4623-4978`: session forking, message-state repointing, and run row mapping.

## Target File Structure

- Create `src/elspeth/web/sessions/_records.py`
  - Owns UTC restoration, JSON envelope helpers, SQLAlchemy row to protocol-record mappers, and ADR-019 legacy run-counter normalization.
  - Contains no database I/O.

- Create `src/elspeth/web/sessions/_interpretation_resolution.py`
  - Owns interpretation-resolution exception classes and pure state-patching helpers.
  - Contains no database transaction management.

- Create `src/elspeth/web/sessions/_base_service.py`
  - Owns `SessionServiceBase`, process-wide SQLite lock globals, `_run_sync`, `_now`, `_ensure_utc`, `_session_write_lock`, `_reserve_sequence_range`, `_insert_chat_message`, and `_insert_composition_state`.
  - Owns the canonical direct insert helpers that static direct-writer tests must recognize.

- Create `src/elspeth/web/sessions/_session_crud.py`
  - Owns `SessionCrudMixin`: `create_session`, `get_session`, `update_session_title`, `list_sessions`, `archive_session`, `get_composer_preferences`, `update_composer_preferences`, and `upsert_skill_markdown_history`.

- Create `src/elspeth/web/sessions/_proposals.py`
  - Owns `ComposerProposalsMixin`: composer proposal create/list/reject/commit and proposal-event listing.

- Create `src/elspeth/web/sessions/_interpretation_events.py`
  - Owns `InterpretationEventsMixin`: `create_pending_interpretation_event`, `resolve_interpretation_event`, `list_interpretation_events`, `record_session_interpretation_opt_out`, and `record_auto_interpreted_no_surfaces_event`.

- Create `src/elspeth/web/sessions/_messages_audit.py`
  - Owns `MessagesAuditMixin`: `add_message`, `get_messages`, `count_tool_responses_for_assistant`, async dispatcher, audit-grade query validation, audit-grade write, async dispatcher, and audit-access-log listing.

- Create `src/elspeth/web/sessions/_state_runs.py`
  - Owns `StateRunMixin`: state save/read/versioning, run lifecycle, blob inline resolution audit rows, active run lookup, active state switching, orphan cancellation, pruning, and message-state repointing.

- Create `src/elspeth/web/sessions/_forking.py`
  - Owns `ForkingMixin`: `fork_session`.

- Modify `src/elspeth/web/sessions/service.py`
  - Imports the mixins and helper modules.
  - Re-exports interpretation exceptions and direct helper names currently imported from `service.py`.
  - Defines only `SessionServiceImpl(...)` as a facade class.

## Task 1: Add A Refactor Structure Canary

**Files:**
- Create: `tests/unit/web/sessions/test_service_refactor_structure.py`
- Modify: none
- Test: `tests/unit/web/sessions/test_service_refactor_structure.py`

- [ ] **Step 1: Write the pending structure canary**

Create `tests/unit/web/sessions/test_service_refactor_structure.py`:

```python
"""Structural canaries for the SessionServiceImpl decomposition.

These tests intentionally pin the refactor shape without changing the
public service API. They fail before the extraction modules exist and
then guard against collapsing the service back into one giant file.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable

import pytest

from elspeth.web.sessions.protocol import SessionServiceProtocol
from elspeth.web.sessions.service import SessionServiceImpl


@pytest.mark.xfail(reason="SessionServiceImpl decomposition modules are introduced across this plan")
def test_session_service_facade_inherits_expected_private_mixins() -> None:
    from elspeth.web.sessions._base_service import SessionServiceBase
    from elspeth.web.sessions._forking import ForkingMixin
    from elspeth.web.sessions._interpretation_events import InterpretationEventsMixin
    from elspeth.web.sessions._messages_audit import MessagesAuditMixin
    from elspeth.web.sessions._proposals import ComposerProposalsMixin
    from elspeth.web.sessions._session_crud import SessionCrudMixin
    from elspeth.web.sessions._state_runs import StateRunMixin

    assert SessionServiceImpl.__mro__[:8] == (
        SessionServiceImpl,
        SessionCrudMixin,
        ComposerProposalsMixin,
        InterpretationEventsMixin,
        MessagesAuditMixin,
        StateRunMixin,
        ForkingMixin,
        SessionServiceBase,
    )


def test_session_service_facade_exposes_protocol_methods() -> None:
    protocol_methods = {
        "create_session",
        "get_session",
        "update_session_title",
        "list_sessions",
        "archive_session",
        "get_composer_preferences",
        "update_composer_preferences",
        "create_composition_proposal",
        "list_composition_proposals",
        "reject_composition_proposal",
        "mark_composition_proposal_committed",
        "list_proposal_events",
        "create_pending_interpretation_event",
        "resolve_interpretation_event",
        "list_interpretation_events",
        "record_session_interpretation_opt_out",
        "upsert_skill_markdown_history",
        "record_auto_interpreted_no_surfaces_event",
        "add_message",
        "get_messages",
        "count_tool_responses_for_assistant_async",
        "record_audit_grade_view_async",
        "save_composition_state",
        "get_current_state",
        "get_state",
        "get_state_in_session",
        "get_state_versions",
        "set_active_state",
        "create_run",
        "get_run",
        "list_runs_for_session",
        "update_run_status",
        "record_blob_inline_resolutions",
        "get_active_run",
        "prune_state_versions",
        "fork_session",
        "update_message_composition_state",
        "cancel_orphaned_runs",
        "cancel_all_orphaned_runs",
        "persist_compose_turn_async",
    }
    for method_name in protocol_methods:
        method = getattr(SessionServiceImpl, method_name)
        assert callable(method), method_name

    assert isinstance(SessionServiceImpl.create_session, Callable)
    assert inspect.isclass(SessionServiceProtocol)
```

- [ ] **Step 2: Run the structure test to verify the canary is pending**

Run:

```bash
.venv/bin/python -m pytest tests/unit/web/sessions/test_service_refactor_structure.py -v
```

Expected: XFAIL with `ModuleNotFoundError: No module named 'elspeth.web.sessions._base_service'`.

- [ ] **Step 3: Commit the pending structure canary**

```bash
git add tests/unit/web/sessions/test_service_refactor_structure.py
git commit -m "test(web): pin session service refactor structure"
```

## Task 2: Extract Pure Record And Envelope Helpers

**Files:**
- Create: `src/elspeth/web/sessions/_records.py`
- Modify: `src/elspeth/web/sessions/service.py`
- Test: `tests/unit/web/sessions/test_datetime_timezone.py`, `tests/unit/web/sessions/test_interpretation_events_service.py`

- [ ] **Step 1: Write the failing direct-module tests**

Append these tests to `tests/unit/web/sessions/test_service_refactor_structure.py`:

```python
from datetime import UTC, datetime
from types import SimpleNamespace
from uuid import uuid4


def test_records_module_restores_sqlite_naive_datetime() -> None:
    from elspeth.web.sessions._records import ensure_utc

    naive = datetime(2026, 6, 4, 10, 20, 30)
    assert ensure_utc(naive) == datetime(2026, 6, 4, 10, 20, 30, tzinfo=UTC)
    assert ensure_utc(datetime(2026, 6, 4, 10, 20, 30, tzinfo=UTC)).tzinfo is UTC


def test_records_module_rejects_empty_interpretation_state_uuid() -> None:
    from elspeth.web.sessions._records import interpretation_event_record_from_row

    row = SimpleNamespace(
        id=str(uuid4()),
        session_id=str(uuid4()),
        composition_state_id="",
        affected_node_id="llm_transform_1",
        tool_call_id="tool-call-abc",
        user_term="term",
        kind="vague_term",
        llm_draft="draft",
        accepted_value="draft",
        choice="accepted_as_drafted",
        created_at=datetime.now(UTC),
        resolved_at=datetime.now(UTC),
        actor="user:alice",
        model_identifier="model",
        model_version="2026-06-04",
        provider="provider",
        composer_skill_hash="a" * 64,
        arguments_hash="b" * 64,
        hash_domain_version="v2",
        interpretation_source="user_approved",
        runtime_model_identifier_at_resolve=None,
        runtime_model_version_at_resolve=None,
        resolved_prompt_template_hash=None,
    )
    try:
        interpretation_event_record_from_row(row)
    except ValueError as exc:
        assert "badly formed" in str(exc) or "hexadecimal" in str(exc) or "UUID" in str(exc)
    else:
        raise AssertionError("empty composition_state_id must reach UUID parsing")
```

- [ ] **Step 2: Run the direct-module tests to verify they fail**

Run:

```bash
.venv/bin/python -m pytest tests/unit/web/sessions/test_service_refactor_structure.py -v -k records
```

Expected: FAIL with `ModuleNotFoundError: No module named 'elspeth.web.sessions._records'`.

- [ ] **Step 3: Create `_records.py` by moving current helper bodies**

Create `src/elspeth/web/sessions/_records.py` with this header and imports:

```python
"""Pure row mappers and JSON envelope helpers for session persistence.

No function in this module opens a database connection or emits telemetry.
Tier-1 persisted rows are converted offensively: malformed UUIDs, unknown
enum values, and invalid envelope versions raise at the conversion site.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from elspeth.contracts.composer_interpretation import (
    InterpretationChoice,
    InterpretationEventRecord,
    InterpretationKind,
    InterpretationSource,
)
from elspeth.contracts.freeze import deep_thaw
from elspeth.web.sessions.protocol import (
    AuditAccessLogRecord,
    ChatMessageRecord,
    CompositionProposalRecord,
    CompositionStateRecord,
    ProposalEventRecord,
    RunRecord,
    SESSION_TERMINAL_RUN_STATUS_VALUES,
    SessionRecord,
    SessionRunStatus,
)
```

Then move these exact existing functions from `src/elspeth/web/sessions/service.py` into `_records.py`, preserving their bodies and comments except for replacing `SessionServiceImpl._ensure_utc(...)` calls with `ensure_utc(...)`:

- `_enveloped_state_column` -> rename to `enveloped_state_column`.
- `_current_adr019_counter_subsets_hold` -> rename to `_current_adr019_counter_subsets_hold`.
- `_legacy_disjoint_counter_shape_holds` -> rename to `_legacy_disjoint_counter_shape_holds`.
- `_normalize_pre_adr019_session_counters` -> rename to `normalize_pre_adr019_session_counters`.
- `_proposal_record_from_row` -> rename to `proposal_record_from_row`.
- `_proposal_event_record_from_row` -> rename to `proposal_event_record_from_row`.
- `_interpretation_event_record_from_row` -> rename to `interpretation_event_record_from_row`.
- `SessionServiceImpl._unwrap_envelope` -> rename to `unwrap_envelope`.
- `SessionServiceImpl._row_to_state_record` -> rename to `row_to_state_record`.
- `SessionServiceImpl._row_to_run_record` -> rename to `row_to_run_record`.

Add these new mappers in `_records.py` using the same field mapping currently repeated in `create_session`, `get_session`, `update_session_title`, and `list_sessions`:

```python
def ensure_utc(dt: datetime) -> datetime:
    """Restore UTC tzinfo stripped by SQLite round-trip."""
    if dt.tzinfo is not None:
        return dt
    return dt.replace(tzinfo=UTC)


def session_record_from_row(row: Any) -> SessionRecord:
    return SessionRecord(
        id=UUID(row.id),
        user_id=row.user_id,
        auth_provider_type=row.auth_provider_type,
        title=row.title,
        created_at=ensure_utc(row.created_at),
        updated_at=ensure_utc(row.updated_at),
        archived_at=ensure_utc(row.archived_at) if row.archived_at else None,
        forked_from_session_id=UUID(row.forked_from_session_id) if row.forked_from_session_id else None,
        forked_from_message_id=UUID(row.forked_from_message_id) if row.forked_from_message_id else None,
    )


def chat_message_record_from_row(row: Any) -> ChatMessageRecord:
    return ChatMessageRecord(
        id=UUID(row.id),
        session_id=UUID(row.session_id),
        role=row.role,
        content=row.content,
        raw_content=row.raw_content,
        tool_calls=row.tool_calls,
        created_at=ensure_utc(row.created_at),
        sequence_no=row.sequence_no,
        composition_state_id=UUID(row.composition_state_id) if row.composition_state_id else None,
        writer_principal=row.writer_principal,
        tool_call_id=row.tool_call_id,
        parent_assistant_id=UUID(row.parent_assistant_id) if row.parent_assistant_id else None,
    )
```

- [ ] **Step 4: Rewire `service.py` to use `_records.py`**

In `src/elspeth/web/sessions/service.py`, add:

```python
from elspeth.web.sessions._records import (
    chat_message_record_from_row,
    ensure_utc,
    enveloped_state_column as _enveloped_state_column,
    interpretation_event_record_from_row as _interpretation_event_record_from_row,
    normalize_pre_adr019_session_counters as _normalize_pre_adr019_session_counters,
    proposal_event_record_from_row as _proposal_event_record_from_row,
    proposal_record_from_row as _proposal_record_from_row,
    row_to_run_record,
    row_to_state_record,
    session_record_from_row,
    unwrap_envelope,
)
```

Replace the duplicate `SessionRecord(...)` and `ChatMessageRecord(...)` constructions in `get_session`, `update_session_title`, `list_sessions`, and `get_messages` with:

```python
return session_record_from_row(row)
```

and:

```python
return [chat_message_record_from_row(row) for row in rows]
```

Keep these compatibility methods on `SessionServiceImpl`:

```python
    @staticmethod
    def _ensure_utc(dt: datetime) -> datetime:
        return ensure_utc(dt)

    @staticmethod
    def _unwrap_envelope(val: Any) -> Any:
        return unwrap_envelope(val)

    def _row_to_state_record(self, row: Any) -> CompositionStateRecord:
        return row_to_state_record(row)

    def _row_to_run_record(self, row: Any) -> RunRecord:
        return row_to_run_record(row)
```

- [ ] **Step 5: Run record and baseline service tests**

Run:

```bash
.venv/bin/python -m pytest \
  tests/unit/web/sessions/test_service_refactor_structure.py \
  tests/unit/web/sessions/test_datetime_timezone.py \
  tests/unit/web/sessions/test_interpretation_events_service.py::test_interpretation_row_conversion_rejects_empty_composition_state_id_as_bad_uuid \
  tests/unit/web/sessions/test_service.py \
  -v
```

Expected: all selected tests PASS; `test_session_service_facade_inherits_expected_private_mixins` remains XFAIL until the mixins exist.

- [ ] **Step 6: Commit**

```bash
git add src/elspeth/web/sessions/_records.py src/elspeth/web/sessions/service.py tests/unit/web/sessions/test_service_refactor_structure.py
git commit -m "refactor(web): extract session row mappers"
```

## Task 3: Extract Pure Interpretation Resolution

**Files:**
- Create: `src/elspeth/web/sessions/_interpretation_resolution.py`
- Modify: `src/elspeth/web/sessions/service.py`
- Modify: `src/elspeth/web/sessions/routes/_helpers.py`
- Modify: `tests/unit/web/sessions/test_interpretation_events_service.py`
- Test: `tests/unit/web/sessions/test_interpretation_events_service.py`

- [ ] **Step 1: Write the failing import test**

Append to `tests/unit/web/sessions/test_service_refactor_structure.py`:

```python
def test_interpretation_resolution_helpers_live_in_private_module() -> None:
    from elspeth.web.sessions._interpretation_resolution import (
        InterpretationPlaceholderConsumedError,
        _patch_llm_transform_prompt,
    )

    assert issubclass(InterpretationPlaceholderConsumedError, ValueError)
    assert callable(_patch_llm_transform_prompt)
```

- [ ] **Step 2: Run the import test to verify it fails**

Run:

```bash
.venv/bin/python -m pytest tests/unit/web/sessions/test_service_refactor_structure.py -v -k interpretation_resolution_helpers
```

Expected: FAIL with `ModuleNotFoundError: No module named 'elspeth.web.sessions._interpretation_resolution'`.

- [ ] **Step 3: Create `_interpretation_resolution.py` by moving the current pure block**

Create `src/elspeth/web/sessions/_interpretation_resolution.py` with this header:

```python
"""Pure interpretation-resolution helpers for session composition state.

The helpers in this module patch in-memory CompositionStateRecord data.
They do not open transactions, write audit rows, emit telemetry, or log.
Service methods own those effects and call these helpers before DB writes.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, TypedDict

from elspeth.contracts.composer_interpretation import (
    INTERPRETATION_HASH_DOMAIN_V2,
    InterpretationKind,
)
from elspeth.contracts.errors import AuditIntegrityError
from elspeth.contracts.hashing import stable_hash
from elspeth.web.interpretation_state import (
    INTERPRETATION_REQUIREMENTS_KEY,
    PENDING_INTERPRETATION_AUTHORING_TEXT,
    PROMPT_TEMPLATE_PARTS_KEY,
    SOURCE_AUTHORING_KEY,
    SOURCE_COMPONENT_ID,
    pipeline_decision_artifact_hash,
    prompt_structure_hash_from_options,
    validate_pipeline_decision_semantics,
)
from elspeth.web.sessions.protocol import CompositionStateRecord
from elspeth.web.validation import INTERPRETATION_PLACEHOLDER_RE
```

Move the exact current block `src/elspeth/web/sessions/service.py:490-1331` into this file, preserving comments and helper bodies. Keep `_STRUCTURAL_DIRECTIVE_PREFIXES` unchanged; the comment says the list is closed and must remain load-bearing.

- [ ] **Step 4: Rewire imports**

In `src/elspeth/web/sessions/service.py`, import the moved names:

```python
from elspeth.web.sessions._interpretation_resolution import (
    InterpretationEventAlreadyResolvedError,
    InterpretationEventNotFoundError,
    InterpretationNodeMissingError,
    InterpretationNodePluginMutatedError,
    InterpretationPlaceholderConsumedError,
    InterpretationResolveError,
    InterpretationUnsupportedChoiceError,
    _interpretation_hash_domain_v2,
    _patch_llm_transform_prompt,
    _resolve_invented_source,
    _resolve_model_choice_review,
    _resolve_pipeline_decision_review,
    _resolve_prompt_template_review,
    _resolve_vague_term,
)
```

In `src/elspeth/web/sessions/routes/_helpers.py`, replace the current error import from `service.py` with:

```python
from elspeth.web.sessions._interpretation_resolution import (
    InterpretationEventAlreadyResolvedError,
    InterpretationEventNotFoundError,
    InterpretationNodeMissingError,
    InterpretationNodePluginMutatedError,
    InterpretationPlaceholderConsumedError,
    InterpretationUnsupportedChoiceError,
)
```

In `tests/unit/web/sessions/test_interpretation_events_service.py`, replace:

```python
from elspeth.web.sessions.service import SessionServiceImpl, _interpretation_event_record_from_row, _patch_llm_transform_prompt
```

with:

```python
from elspeth.web.sessions._interpretation_resolution import _patch_llm_transform_prompt
from elspeth.web.sessions._records import interpretation_event_record_from_row as _interpretation_event_record_from_row
from elspeth.web.sessions.service import SessionServiceImpl
```

- [ ] **Step 5: Run interpretation tests**

Run:

```bash
.venv/bin/python -m pytest tests/unit/web/sessions/test_interpretation_events_service.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add \
  src/elspeth/web/sessions/_interpretation_resolution.py \
  src/elspeth/web/sessions/service.py \
  src/elspeth/web/sessions/routes/_helpers.py \
  tests/unit/web/sessions/test_interpretation_events_service.py \
  tests/unit/web/sessions/test_service_refactor_structure.py
git commit -m "refactor(web): extract interpretation resolution helpers"
```

## Task 4: Extract The Base Service And Write Coordination

**Files:**
- Create: `src/elspeth/web/sessions/_base_service.py`
- Modify: `src/elspeth/web/sessions/service.py`
- Modify: `tests/unit/web/sessions/test_static_direct_writers.py`
- Test: `tests/unit/web/sessions/test_persist_compose_turn.py`, `tests/unit/web/sessions/test_static_direct_writers.py`

- [ ] **Step 1: Run the structure test to confirm the base module is still missing**

Run:

```bash
.venv/bin/python -m pytest tests/unit/web/sessions/test_service_refactor_structure.py -v -k facade_inherits
```

Expected: XFAIL while later mixin modules are still missing.

- [ ] **Step 2: Create `_base_service.py`**

Create `src/elspeth/web/sessions/_base_service.py` with this header and imports:

```python
"""Shared base class and write coordination for SessionServiceImpl mixins."""

from __future__ import annotations

import contextlib
import threading
import uuid
from collections.abc import Iterator, Mapping
from contextvars import ContextVar
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import structlog
from sqlalchemy import Connection, Engine, func, insert, select

from elspeth.contracts.advisory_locks import ELSPETH_SESSIONS_LOCK_CLASSID
from elspeth.contracts.freeze import deep_thaw
from elspeth.web.async_workers import run_sync_in_worker
from elspeth.web.sessions._persist_payload import StatePayload
from elspeth.web.sessions._records import enveloped_state_column
from elspeth.web.sessions.models import chat_messages_table, composition_states_table
from elspeth.web.sessions.protocol import ChatMessageWriterPrincipal, ToolCallIDMismatchError
from elspeth.web.sessions.telemetry import _SessionsTelemetry
```

Move these exact current definitions from `service.py` into `_base_service.py`:

- `_SQLITE_SESSION_LOCKS_GUARD`
- `_SQLITE_SESSION_LOCKS`
- `_SESSION_WRITE_LOCK_HELD`
- `_assert_state_in_session`
- `_assert_parent_assistant_message`
- `_validate_tool_call_id_set_equality`
- `SessionServiceImpl.__init__`
- `SessionServiceImpl._run_sync`
- `SessionServiceImpl._now`
- `SessionServiceImpl._ensure_utc`
- `SessionServiceImpl._acquire_session_advisory_lock`
- `SessionServiceImpl._sqlite_lock_for_session`
- `SessionServiceImpl._assert_session_write_lock_held`
- `SessionServiceImpl._session_write_lock`
- `SessionServiceImpl._reserve_sequence_range`
- `SessionServiceImpl._insert_chat_message`
- `SessionServiceImpl._insert_composition_state`

Wrap the moved methods in:

```python
class SessionServiceBase:
    """Shared infrastructure for the private session service mixins."""
```

Inside `_insert_composition_state`, replace `_enveloped_state_column(...)` with `enveloped_state_column(...)`.

- [ ] **Step 3: Rewire `service.py` to inherit the base**

Add:

```python
from elspeth.web.sessions._base_service import (
    SessionServiceBase,
    _assert_state_in_session,
    _validate_tool_call_id_set_equality,
)
```

Change the class line temporarily to:

```python
class SessionServiceImpl(SessionServiceBase):
```

Do not move `persist_compose_turn` yet; it can remain on `SessionServiceImpl` while using inherited helper methods.

- [ ] **Step 4: Update static direct-writer allowlist entries**

In `tests/unit/web/sessions/test_static_direct_writers.py`, change the reviewed writer entries for helper writers:

```python
ReviewedWriter(
    path="src/elspeth/web/sessions/_base_service.py",
    enclosing_symbol="SessionServiceBase._insert_chat_message",
    table="chat_messages",
    operation="sqlalchemy_insert_call",
    purpose=(
        "Canonical chat_messages writer moved from SessionServiceImpl to "
        "SessionServiceBase during the 2026-06-04 service decomposition. "
        "Caller is still required to be inside _session_write_lock and to "
        "reserve sequence_no through _reserve_sequence_range first."
    ),
),
ReviewedWriter(
    path="src/elspeth/web/sessions/_base_service.py",
    enclosing_symbol="SessionServiceBase._insert_composition_state",
    table="composition_states",
    operation="sqlalchemy_insert_call",
    purpose=(
        "Canonical composition_states writer moved from SessionServiceImpl to "
        "SessionServiceBase during the 2026-06-04 service decomposition. "
        "Version allocation remains inside the held _session_write_lock."
    ),
),
```

Also update any lock-discipline negative-test entries that key on `SessionServiceImpl._insert_chat_message` or `SessionServiceImpl._insert_composition_state` so their `enclosing_symbol` values use `SessionServiceBase`.

- [ ] **Step 5: Run persistence and static guard tests**

Run:

```bash
.venv/bin/python -m pytest \
  tests/unit/web/sessions/test_persist_compose_turn.py \
  tests/unit/web/sessions/test_static_direct_writers.py \
  tests/unit/web/sessions/test_service_construction.py \
  -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add \
  src/elspeth/web/sessions/_base_service.py \
  src/elspeth/web/sessions/service.py \
  tests/unit/web/sessions/test_static_direct_writers.py
git commit -m "refactor(web): extract session service write coordination"
```

## Task 5: Extract Session CRUD And Preferences

**Files:**
- Create: `src/elspeth/web/sessions/_session_crud.py`
- Modify: `src/elspeth/web/sessions/service.py`
- Test: `tests/unit/web/sessions/test_service.py`, `tests/unit/web/sessions/test_routes.py`

- [ ] **Step 1: Create the mixin module**

Create `src/elspeth/web/sessions/_session_crud.py` with:

```python
"""Session CRUD, archival, composer preference, and skill-history methods."""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path
from typing import Any, cast
from uuid import UUID

from sqlalchemy import ColumnElement, delete, desc, insert, select, update
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from elspeth.contracts.auth import AuthProviderType
from elspeth.web.sessions._records import session_record_from_row
from elspeth.web.sessions.models import (
    composer_completion_events_table,
    proposal_events_table,
    runs_table,
    sessions_table,
    skill_markdown_history_table,
)
from elspeth.web.sessions.protocol import (
    ComposerDensityDefault,
    ComposerSessionPreferencesRecord,
    ComposerSessionPreferencesTransition,
    ComposerTrustMode,
    SessionNotFoundError,
    SessionRecord,
)


class SessionCrudMixin:
    """Session row CRUD and preference methods for SessionServiceImpl."""
```

Move these exact methods from `service.py` into `SessionCrudMixin`:

- `create_session`
- `get_session`
- `update_session_title`
- `list_sessions`
- `archive_session`
- `get_composer_preferences`
- `update_composer_preferences`
- `upsert_skill_markdown_history`

Use `session_record_from_row(row)` for session row conversion.

- [ ] **Step 2: Rewire the facade**

In `src/elspeth/web/sessions/service.py`, import the mixin:

```python
from elspeth.web.sessions._session_crud import SessionCrudMixin
```

Temporarily change the class line to:

```python
class SessionServiceImpl(SessionCrudMixin, SessionServiceBase):
```

- [ ] **Step 3: Run CRUD and route tests**

Run:

```bash
.venv/bin/python -m pytest \
  tests/unit/web/sessions/test_service.py \
  tests/unit/web/sessions/test_routes.py -k "session or preferences or archive" \
  tests/unit/web/sessions/test_service_refactor_structure.py -k "facade_exposes" \
  -v
```

Expected: PASS for selected behavior tests; the exact MRO structure test remains XFAIL until every mixin is extracted and the final class order is set.

- [ ] **Step 4: Commit**

```bash
git add src/elspeth/web/sessions/_session_crud.py src/elspeth/web/sessions/service.py
git commit -m "refactor(web): extract session crud mixin"
```

## Task 6: Extract Composer Proposal Lifecycle

**Files:**
- Create: `src/elspeth/web/sessions/_proposals.py`
- Modify: `src/elspeth/web/sessions/service.py`
- Test: `tests/unit/web/sessions/test_composer_proposals.py`

- [ ] **Step 1: Create the proposal mixin**

Create `src/elspeth/web/sessions/_proposals.py` with:

```python
"""Composer proposal lifecycle methods for SessionServiceImpl."""

from __future__ import annotations

import uuid
from collections.abc import Mapping, Sequence
from typing import Any, cast
from uuid import UUID

from sqlalchemy import insert, select, update

from elspeth.contracts.errors import AuditIntegrityError
from elspeth.contracts.freeze import deep_thaw
from elspeth.web.sessions._records import proposal_event_record_from_row, proposal_record_from_row
from elspeth.web.sessions.models import composition_proposals_table, proposal_events_table
from elspeth.web.sessions.protocol import CompositionProposalRecord, ProposalEventRecord, ProposalLifecycleStatus

_PROPOSAL_COMPOSER_PROVENANCE_FIELDS = (
    "composer_model_identifier",
    "composer_model_version",
    "composer_provider",
    "composer_skill_hash",
    "tool_arguments_hash",
)
```

Move these exact current helpers and methods into `_proposals.py`:

- `_normalize_optional_provenance_text`
- `_normalize_proposal_composer_provenance`
- `create_composition_proposal`
- `list_composition_proposals`
- `reject_composition_proposal`
- `mark_composition_proposal_committed`
- `list_proposal_events`

Wrap the moved methods in:

```python
class ComposerProposalsMixin:
    """Composer proposal create/list/transition methods."""
```

- [ ] **Step 2: Rewire the facade**

In `service.py`, import:

```python
from elspeth.web.sessions._proposals import ComposerProposalsMixin
```

Change the class line to:

```python
class SessionServiceImpl(SessionCrudMixin, ComposerProposalsMixin, SessionServiceBase):
```

- [ ] **Step 3: Run proposal tests**

Run:

```bash
.venv/bin/python -m pytest tests/unit/web/sessions/test_composer_proposals.py -v
```

Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add src/elspeth/web/sessions/_proposals.py src/elspeth/web/sessions/service.py
git commit -m "refactor(web): extract composer proposal mixin"
```

## Task 7: Extract Interpretation Event Methods

**Files:**
- Create: `src/elspeth/web/sessions/_interpretation_events.py`
- Modify: `src/elspeth/web/sessions/service.py`
- Test: `tests/unit/web/sessions/test_interpretation_events_service.py`, `tests/unit/web/sessions/test_interpretation_events_routes.py`

- [ ] **Step 1: Create the interpretation-event mixin**

Create `src/elspeth/web/sessions/_interpretation_events.py` with:

```python
"""Interpretation event persistence methods for SessionServiceImpl."""

from __future__ import annotations

import uuid
from collections.abc import Mapping, Sequence
from dataclasses import replace
from datetime import datetime
from typing import Literal, cast
from uuid import UUID

from sqlalchemy import desc, insert, select, update
from sqlalchemy.exc import IntegrityError

from elspeth.contracts.composer_interpretation import InterpretationChoice, InterpretationEventRecord, InterpretationKind, InterpretationSource
from elspeth.contracts.errors import AuditIntegrityError
from elspeth.contracts.hashing import stable_hash
from elspeth.web.composer.telemetry_phase8 import record_interpretation_opt_out
from elspeth.web.interpretation_state import (
    INTERPRETATION_REQUIREMENTS_KEY,
    SOURCE_AUTHORING_KEY,
    SOURCE_COMPONENT_ID,
    validate_pipeline_decision_semantics,
)
from elspeth.web.sessions._interpretation_resolution import (
    InterpretationEventAlreadyResolvedError,
    InterpretationEventNotFoundError,
    InterpretationPlaceholderConsumedError,
    InterpretationUnsupportedChoiceError,
    _find_interpretation_review_node,
    _find_llm_transform_node,
    _interpretation_hash_domain_v2,
    _matching_pending_requirement_index,
    _require_mapping,
    _resolve_invented_source,
    _resolve_model_choice_review,
    _resolve_pipeline_decision_review,
    _resolve_prompt_template_review,
    _resolve_vague_term,
)
from elspeth.web.sessions._persist_payload import StatePayload
from elspeth.web.sessions._records import interpretation_event_record_from_row
from elspeth.web.sessions.models import composition_states_table, interpretation_events_table, sessions_table
from elspeth.web.sessions.protocol import CompositionStateData, CompositionStateRecord
from elspeth.web.validation import _validate_accepted_value_content

_INTERPRETATION_IMMUTABLE_TRIGGER_MSG: str = "interpretation_events: resolved rows are immutable"


class InterpretationEventsMixin:
    """Interpretation-event writer and reader methods."""
```

Move these exact methods from `service.py` into `InterpretationEventsMixin`:

- `create_pending_interpretation_event`
- `resolve_interpretation_event`
- `list_interpretation_events`
- `record_session_interpretation_opt_out`
- `record_auto_interpreted_no_surfaces_event`

Keep `_INTERPRETATION_IMMUTABLE_TRIGGER_MSG` in this module.

- [ ] **Step 2: Rewire the facade**

In `service.py`, import:

```python
from elspeth.web.sessions._interpretation_events import InterpretationEventsMixin
```

Change the class line to:

```python
class SessionServiceImpl(
    SessionCrudMixin,
    ComposerProposalsMixin,
    InterpretationEventsMixin,
    SessionServiceBase,
):
```

- [ ] **Step 3: Run interpretation tests**

Run:

```bash
.venv/bin/python -m pytest \
  tests/unit/web/sessions/test_interpretation_events_service.py \
  tests/unit/web/sessions/test_interpretation_events_routes.py \
  tests/unit/web/composer/test_request_interpretation_review_tool.py \
  -v
```

Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add src/elspeth/web/sessions/_interpretation_events.py src/elspeth/web/sessions/service.py
git commit -m "refactor(web): extract interpretation event mixin"
```

## Task 8: Extract Messages And Audit Access Logging

**Files:**
- Create: `src/elspeth/web/sessions/_messages_audit.py`
- Modify: `src/elspeth/web/sessions/service.py`
- Test: `tests/unit/web/sessions/test_count_tool_responses_for_assistant.py`, `tests/unit/web/sessions/test_record_audit_grade_view.py`, `tests/unit/web/sessions/test_chat_messages.py`

- [ ] **Step 1: Create the messages/audit mixin**

Create `src/elspeth/web/sessions/_messages_audit.py` with:

```python
"""Chat-message and audit-access-log methods for SessionServiceImpl."""

from __future__ import annotations

import uuid
from collections.abc import Mapping, Sequence
from typing import Any, cast
from uuid import UUID

from sqlalchemy import func, select, update
from sqlalchemy.exc import SQLAlchemyError

from elspeth.contracts.freeze import deep_thaw
from elspeth.web.sessions._base_service import _assert_state_in_session
from elspeth.web.sessions._records import chat_message_record_from_row
from elspeth.web.sessions.models import audit_access_log_table, chat_messages_table, sessions_table
from elspeth.web.sessions.protocol import (
    AUDIT_GRADE_VIEW_QUERY_ARG_ALLOWLIST,
    AUDIT_GRADE_VIEW_WRITER_PRINCIPAL,
    AuditAccessLogRecord,
    AuditAccessLogWriteError,
    ChatMessageRecord,
    ChatMessageRole,
    ChatMessageWriterPrincipal,
)


class MessagesAuditMixin:
    """Chat messages and audit-grade access logging."""
```

Move these exact methods from `service.py` into `MessagesAuditMixin`:

- `add_message`
- `get_messages`
- `count_tool_responses_for_assistant`
- `count_tool_responses_for_assistant_async`
- `_validate_audit_grade_query_args`
- `record_audit_grade_view`
- `record_audit_grade_view_async`
- `list_audit_access_log`

Use `chat_message_record_from_row(row)` in `get_messages`.

- [ ] **Step 2: Rewire the facade**

In `service.py`, import:

```python
from elspeth.web.sessions._messages_audit import MessagesAuditMixin
```

Change the class line to:

```python
class SessionServiceImpl(
    SessionCrudMixin,
    ComposerProposalsMixin,
    InterpretationEventsMixin,
    MessagesAuditMixin,
    SessionServiceBase,
):
```

- [ ] **Step 3: Update static direct-writer allowlist for `add_message` comments only**

In `tests/unit/web/sessions/test_static_direct_writers.py`, update the comment that names `SessionServiceImpl.add_message._sync` so it names `MessagesAuditMixin.add_message._sync`. No reviewed writer entry should be added for `add_message` because the method must still route through `_insert_chat_message`.

- [ ] **Step 4: Run message and audit tests**

Run:

```bash
.venv/bin/python -m pytest \
  tests/unit/web/sessions/test_count_tool_responses_for_assistant.py \
  tests/unit/web/sessions/test_record_audit_grade_view.py \
  tests/unit/web/sessions/test_chat_messages.py \
  tests/unit/web/sessions/test_static_direct_writers.py \
  -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add \
  src/elspeth/web/sessions/_messages_audit.py \
  src/elspeth/web/sessions/service.py \
  tests/unit/web/sessions/test_static_direct_writers.py
git commit -m "refactor(web): extract session messages and audit access"
```

## Task 9: Extract State And Run Methods

**Files:**
- Create: `src/elspeth/web/sessions/_state_runs.py`
- Modify: `src/elspeth/web/sessions/service.py`
- Modify: `tests/unit/web/sessions/test_static_direct_writers.py`
- Test: `tests/unit/web/sessions/test_service.py`, `tests/unit/web/sessions/test_record_blob_inline_resolutions.py`, `tests/unit/web/sessions/test_static_direct_writers.py`

- [ ] **Step 1: Create the state/run mixin**

Create `src/elspeth/web/sessions/_state_runs.py` with:

```python
"""Composition-state and run lifecycle methods for SessionServiceImpl."""

from __future__ import annotations

import uuid
from datetime import timedelta
from typing import Any, cast
from uuid import UUID

from sqlalchemy import delete, desc, func, insert, select, update
from sqlalchemy.exc import IntegrityError, SQLAlchemyError

from elspeth.contracts.blobs_inline import ResolvedBlobContent
from elspeth.contracts.errors import AuditIntegrityError
from elspeth.contracts.freeze import deep_thaw
from elspeth.web.sessions._base_service import _assert_state_in_session
from elspeth.web.sessions._persist_payload import StatePayload
from elspeth.web.sessions._records import enveloped_state_column, row_to_run_record, row_to_state_record, unwrap_envelope
from elspeth.web.sessions.models import (
    blob_inline_resolutions_table,
    chat_messages_table,
    composition_states_table,
    runs_table,
)
from elspeth.web.sessions.protocol import (
    LEGAL_RUN_TRANSITIONS,
    OPERATOR_COMPLETION_RUN_STATUS_VALUES,
    SESSION_TERMINAL_RUN_STATUS_VALUES,
    CompositionStateData,
    CompositionStateProvenance,
    CompositionStateRecord,
    IllegalRunTransitionError,
    RunAlreadyActiveError,
    RunRecord,
    SessionRunStatus,
)
```

Move these exact methods from `service.py` into `StateRunMixin`:

- `save_composition_state`
- `get_current_state`
- `get_state_versions`
- `_unwrap_envelope`
- `_row_to_state_record`
- `create_run`
- `get_run`
- `list_runs_for_session`
- `update_run_status`
- `record_blob_inline_resolutions`
- `get_active_run`
- `get_state`
- `get_state_in_session`
- `set_active_state`
- `cancel_orphaned_runs`
- `cancel_all_orphaned_runs`
- `prune_state_versions`
- `update_message_composition_state`
- `_row_to_run_record`

Keep `_unwrap_envelope`, `_row_to_state_record`, and `_row_to_run_record` as thin delegates:

```python
    @staticmethod
    def _unwrap_envelope(val: Any) -> Any:
        return unwrap_envelope(val)

    def _row_to_state_record(self, row: Any) -> CompositionStateRecord:
        return row_to_state_record(row)

    def _row_to_run_record(self, row: Any) -> RunRecord:
        return row_to_run_record(row)
```

- [ ] **Step 2: Rewire the facade**

In `service.py`, import:

```python
from elspeth.web.sessions._state_runs import StateRunMixin
```

Change the class line to:

```python
class SessionServiceImpl(
    SessionCrudMixin,
    ComposerProposalsMixin,
    InterpretationEventsMixin,
    MessagesAuditMixin,
    StateRunMixin,
    SessionServiceBase,
):
```

- [ ] **Step 3: Update static direct-writer allowlist for moved inline state writers**

In `tests/unit/web/sessions/test_static_direct_writers.py`, change the `ReviewedWriter` entries for inline `composition_states` writers:

```python
ReviewedWriter(
    path="src/elspeth/web/sessions/_state_runs.py",
    enclosing_symbol="StateRunMixin.save_composition_state._sync",
    table="composition_states",
    operation="sqlalchemy_insert_call",
    purpose=(
        "save_composition_state inline writer moved from service.py to "
        "_state_runs.py during the 2026-06-04 service decomposition. "
        "The SELECT-MAX + INSERT region remains wrapped in _session_write_lock."
    ),
),
ReviewedWriter(
    path="src/elspeth/web/sessions/_state_runs.py",
    enclosing_symbol="StateRunMixin.set_active_state._sync",
    table="composition_states",
    operation="sqlalchemy_insert_call",
    purpose=(
        "set_active_state inline writer moved from service.py to "
        "_state_runs.py during the 2026-06-04 service decomposition. "
        "The copy-as-new-version transaction remains under _session_write_lock."
    ),
),
```

- [ ] **Step 4: Run state/run tests**

Run:

```bash
.venv/bin/python -m pytest \
  tests/unit/web/sessions/test_service.py \
  tests/unit/web/sessions/test_record_blob_inline_resolutions.py \
  tests/unit/web/sessions/test_static_direct_writers.py \
  tests/unit/web/sessions/test_routes.py -k "run or state" \
  -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add \
  src/elspeth/web/sessions/_state_runs.py \
  src/elspeth/web/sessions/service.py \
  tests/unit/web/sessions/test_static_direct_writers.py
git commit -m "refactor(web): extract state and run mixin"
```

## Task 10: Extract Forking

**Files:**
- Create: `src/elspeth/web/sessions/_forking.py`
- Modify: `src/elspeth/web/sessions/service.py`
- Modify: `tests/unit/web/sessions/test_static_direct_writers.py`
- Test: `tests/unit/web/sessions/test_fork.py`, `tests/unit/web/sessions/test_static_direct_writers.py`

- [ ] **Step 1: Create the forking mixin**

Create `src/elspeth/web/sessions/_forking.py` with:

```python
"""Session forking methods for SessionServiceImpl."""

from __future__ import annotations

import uuid
from typing import Any
from uuid import UUID

from sqlalchemy import insert

from elspeth.contracts.auth import AuthProviderType
from elspeth.contracts.freeze import deep_thaw
from elspeth.web.sessions._persist_payload import StatePayload
from elspeth.web.sessions.models import chat_messages_table, sessions_table
from elspeth.web.sessions.protocol import ChatMessageRecord, CompositionStateData, CompositionStateRecord, InvalidForkTargetError, SessionRecord


class ForkingMixin:
    """Session fork writer."""
```

Move `fork_session` from `service.py` into `ForkingMixin`, preserving its body and comments.

- [ ] **Step 2: Rewire the facade**

In `service.py`, import:

```python
from elspeth.web.sessions._forking import ForkingMixin
```

Set the final class line to:

```python
class SessionServiceImpl(
    SessionCrudMixin,
    ComposerProposalsMixin,
    InterpretationEventsMixin,
    MessagesAuditMixin,
    StateRunMixin,
    ForkingMixin,
    SessionServiceBase,
):
    """Concrete session service backed by SQLAlchemy Core.

    All public methods are async. Database I/O runs through _run_sync() in a
    bounded worker thread so the async event loop is never blocked.
    """
```

- [ ] **Step 3: Update static direct-writer allowlist for fork batch writer**

In `tests/unit/web/sessions/test_static_direct_writers.py`, change the fork writer entry:

```python
ReviewedWriter(
    path="src/elspeth/web/sessions/_forking.py",
    enclosing_symbol="ForkingMixin.fork_session._sync",
    table="chat_messages",
    operation="sqlalchemy_insert_call",
    purpose=(
        "fork_session batch-copies source-session chat rows. The method moved "
        "from service.py to _forking.py during the 2026-06-04 service "
        "decomposition; it still reserves sequence_no under _session_write_lock "
        "and rewrites copied tool parent_assistant_id values before the batch insert."
    ),
),
```

- [ ] **Step 4: Run fork and structure tests**

Before running the structure test, remove this marker from `tests/unit/web/sessions/test_service_refactor_structure.py`:

```python
@pytest.mark.xfail(reason="SessionServiceImpl decomposition modules are introduced across this plan")
```

If `pytest` is no longer imported after removing the marker, remove the `import pytest` line too.

Run:

```bash
.venv/bin/python -m pytest \
  tests/unit/web/sessions/test_fork.py \
  tests/unit/web/sessions/test_static_direct_writers.py \
  tests/unit/web/sessions/test_service_refactor_structure.py \
  -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add \
  src/elspeth/web/sessions/_forking.py \
  src/elspeth/web/sessions/service.py \
  tests/unit/web/sessions/test_static_direct_writers.py
git commit -m "refactor(web): extract session forking mixin"
```

## Task 11: Move Compose-Turn Persistence Into The Base Module

**Files:**
- Modify: `src/elspeth/web/sessions/_base_service.py`
- Modify: `src/elspeth/web/sessions/service.py`
- Test: `tests/unit/web/sessions/test_persist_compose_turn.py`, `tests/integration/web/test_compose_loop_concurrent_sessions.py`

- [ ] **Step 1: Move `persist_compose_turn` and dispatcher**

Move these exact methods from `SessionServiceImpl` in `service.py` into `SessionServiceBase` in `_base_service.py`:

- `persist_compose_turn`
- `persist_compose_turn_async`

Add imports to `_base_service.py`:

```python
from sqlalchemy.exc import IntegrityError, OperationalError, SQLAlchemyError

from elspeth.contracts.errors import AuditIntegrityError
from elspeth.web.sessions._persist_payload import AuditOutcome, RedactedToolRow, StatePayload
from elspeth.web.sessions.protocol import StaleComposeStateError
```

Keep the async-loop guard, audit failure disposition, telemetry counters, and `plugin_crash_pending` asymmetry unchanged.

- [ ] **Step 2: Confirm `service.py` facade has no method bodies**

After this move, `src/elspeth/web/sessions/service.py` should contain only imports, re-exported imported names, and the `SessionServiceImpl` class definition. Verify:

```bash
rg -n "^    (async def|def) " src/elspeth/web/sessions/service.py
```

Expected: no output.

- [ ] **Step 3: Run compose-turn tests**

Run:

```bash
.venv/bin/python -m pytest \
  tests/unit/web/sessions/test_persist_compose_turn.py \
  tests/integration/web/test_compose_loop_concurrent_sessions.py \
  -v
```

Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add src/elspeth/web/sessions/_base_service.py src/elspeth/web/sessions/service.py
git commit -m "refactor(web): move compose turn persistence to base service"
```

## Task 12: Final Verification And Tier-Lint Sweep

**Files:**
- Modify only if tests expose stale import paths or allowlist entries.
- Test: full focused sessions/composer backend suite plus tier lint.

- [ ] **Step 1: Run focused sessions suite**

Run:

```bash
.venv/bin/python -m pytest tests/unit/web/sessions -v
```

Expected: PASS.

- [ ] **Step 2: Run focused composer/session integration sweep**

Run:

```bash
.venv/bin/python -m pytest \
  tests/unit/web/composer/test_request_interpretation_review_tool.py \
  tests/unit/web/composer/test_compose_loop_interpretation_review_dispatch.py \
  tests/integration/web/test_compose_loop_concurrent_sessions.py \
  tests/integration/web/composer/test_interpretation_runtime_handoff.py \
  -v
```

Expected: PASS.

- [ ] **Step 3: Run static direct-writer guard**

Run:

```bash
.venv/bin/python -m pytest tests/unit/web/sessions/test_static_direct_writers.py -v
```

Expected: PASS.

- [ ] **Step 4: Run trust-tier lint**

Run:

```bash
env PYTHONPATH=elspeth-lints/src .venv/bin/python -m elspeth_lints.core.cli check \
  --rules trust_tier.tier_model \
  --root src/elspeth
```

Expected: PASS with no new unsuppressed tier-model findings. If the move changes a signed allowlist scope, stop and surface the exact finding. Do not hand-edit judge metadata; operator-only signed allowlist repair is required for HMAC-bound entries.

- [ ] **Step 5: Run import smoke**

Run:

```bash
.venv/bin/python - <<'PY'
from elspeth.web.sessions.service import (
    InterpretationEventAlreadyResolvedError,
    InterpretationEventNotFoundError,
    InterpretationNodeMissingError,
    InterpretationNodePluginMutatedError,
    InterpretationPlaceholderConsumedError,
    InterpretationUnsupportedChoiceError,
    SessionServiceImpl,
)

print(SessionServiceImpl.__mro__)
print(InterpretationEventNotFoundError.__name__)
print(InterpretationEventAlreadyResolvedError.__name__)
print(InterpretationNodeMissingError.__name__)
print(InterpretationNodePluginMutatedError.__name__)
print(InterpretationPlaceholderConsumedError.__name__)
print(InterpretationUnsupportedChoiceError.__name__)
PY
```

Expected: command exits 0 and prints `SessionServiceImpl` MRO plus six exception class names.

- [ ] **Step 6: Commit verification-only fixes if any were needed**

If Step 1-5 exposed stale import paths or static guard entries, commit those exact fixes:

```bash
git add src/elspeth/web/sessions tests/unit/web/sessions tests/unit/web/composer tests/integration/web
git commit -m "test(web): refresh session service refactor guards"
```

If no fixes were needed, do not create an empty commit.

## Rollback Strategy

Every task is a behavior-preserving move with a focused commit. If any task fails and the fix is not obvious from the failing import or allowlist entry, stop at that task's commit boundary and inspect the moved module's imports first. Do not use `git reset --hard`, `git clean`, or checkout-discard commands without explicit operator permission.

## Acceptance Criteria

- `src/elspeth/web/sessions/service.py` contains no method bodies and only defines the facade class plus imports/re-exports.
- `SessionServiceImpl(...)` still constructs with `(engine, data_dir=None, *, telemetry, log)`.
- All methods in `SessionServiceProtocol` remain callable on `SessionServiceImpl`.
- `tests/unit/web/sessions/test_service_refactor_structure.py` passes.
- `tests/unit/web/sessions/test_static_direct_writers.py` passes with reviewed entries pointing at the new files.
- Interpretation event route error mapping still imports the same exception classes or their new private-module source.
- No telemetry/logging is added to interpretation event methods.
- Trust-tier lint reports no new unsuppressed findings.
