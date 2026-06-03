# Composer Progress Persistence — Phase 1: Data Layer + Sync Primitive

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the schema columns, helper functions, and synchronous persistence primitive (`SessionServiceImpl.persist_compose_turn`) that subsequent phases need. End state: every existing `add_message` caller passes `writer_principal`; the database refuses unrecognised principals; audit-only composer breadcrumbs have a schema-compatible storage shape; same-session sequence/version allocation is serialized on both SQLite and PostgreSQL; backward-direction INV-AUDIT-AHEAD is provable from schema state.

**Architecture:** Pure data-layer work. No compose-loop changes; no redaction primitives; no frontend. The new primitive is dispatched via the existing `_run_sync` helper; current code creates a one-shot worker executor per call, so this plan must not describe a shared worker pool unless it introduces one explicitly. Tests use in-memory SQLite through `create_session_engine(..., StaticPool)` + `initialize_session_schema()` for schema/unit work and testcontainer PostgreSQL for the advisory-lock and concurrent-multi-session test (CL-PP-11).

**Tech Stack:** Python 3.12 and 3.13 (matching supported CI), SQLAlchemy 2.x sync `Engine`, structlog, OpenTelemetry counters, pytest, testcontainers-python (PostgreSQL).

**Spec sections:** §3 (ADRs), §4.1 (schema), §4.5 (IntegrityError + OperationalError dispositions), §5.7 (SessionServiceImpl API), §8.1 unit tests bullets 1–3 + 5–7, §8.6 test-path-integrity rule, §11 Phase 1 scope.

---

## Risk-First Execution Schedules

**Review status:** CHANGES_REQUESTED. Do not execute this document as one
large Phase 1 PR. The task bodies below remain the detailed
implementation source, but execution is split into three schedules with
separate review gates. The split optimizes for risk containment, not
calendar speed.

**Scheduling rule:** only one schedule is active at a time. Schedule B
does not start until Schedule A has passed review and landed; Schedule C
does not start until Schedule B has passed review and landed. If an
implementer finds a blocker that belongs to a later schedule, they file
it against that later schedule instead of broadening the active PR.

**Canonical review files:**
- Schedule A: `docs/superpowers/plans/completed/composer-progress-persistence/2026-04-30-composer-progress-persistence-phase-1A-schema-current-writer-safety.md`
- Schedule B: `docs/superpowers/plans/completed/composer-progress-persistence/2026-04-30-composer-progress-persistence-phase-1B-compose-turn-primitive-audit-semantics.md`
- Schedule C: `docs/superpowers/plans/completed/composer-progress-persistence/2026-04-30-composer-progress-persistence-phase-1C-postgresql-ci-operational-proof.md`

Use those files for agent review. This parent file remains the
traceability source for the original full Phase 1 task numbering.

### Schedule A: Schema and Current Writer Safety

**Risk controlled:** destructive schema changes, stale staging DB
bootstrap, and regressions in existing message/state writers.

**Scope:**
- Task 0 spec supersession marker, plus blocker-ID and stale-anchor
  cleanup before code execution.
- A new preflight direct-write inventory for every
  `chat_messages_table` and `composition_states_table` writer across
  `src/` and `tests/`; this inventory is a merge gate.
- Task 1 through Task 4 schema work, after fixing plan-authored import
  errors and PostgreSQL-incompatible metadata notes called out by the
  review.
- Task 7 through Task 10 only where needed for current writer safety:
  `_session_write_lock`, sequence/version allocation helpers,
  composition-state provenance, and stale-current-state guards.
- Task 14 current writer cutover: `add_message`, `get_messages`
  ordering/filtering, `fork_session`, route call sites, protocol record
  hydration, direct test fixtures, and the missing `ChatMessageRole`
  import.
- Task 18 moved into this schedule before the schema PR lands:
  staging session-DB recreation runbook plus live DB dialect/path
  inspection.

**Out of scope:**
- `persist_compose_turn` and `persist_compose_turn_async`.
- Compose-turn telemetry/payload dataclasses unless a current-writer
  compatibility test proves they are required.
- PostgreSQL concurrency claims beyond schema-portability smoke checks.

**Exit gates:**
- No direct writer can insert a chat message or composition state
  without satisfying the new required columns.
- The staging DB recreation runbook exists and is explicitly reviewed
  before any schema-breaking deploy.
- SQLite unit/integration tests for current behavior pass.
- A follow-up review confirms B2, B3 import/precondition items, B8,
  B9, W1, W2, W3, and W5 are either resolved or explicitly carried
  into Schedule B/C with a non-blocking rationale.

### Schedule B: Compose-Turn Primitive and Audit Semantics

**Risk controlled:** atomic compose-turn persistence, audit primacy,
tool-call transcript consistency, and async cancellation/idempotency
before any production route depends on the primitive.

**Scope:**
- Task 5 and Task 6 telemetry/payload dataclasses, unless Schedule A
  deliberately pulled a minimal subset forward.
- Task 11 through Task 13: `persist_compose_turn`,
  `persist_compose_turn_async`, `_AuditOutcome`, unwind behavior, and
  failure disposition.
- Exact validation that assistant `tool_calls` match redacted tool rows
  before any insert, including missing, extra, mismatched, and duplicate
  ID regressions with no partial writes.
- A cancellation/idempotency contract for the shielded worker bridge,
  including retry-after-cancel behavior.
- Task 15 backward-direction INV-AUDIT-AHEAD proof where it can be
  exercised without relying on PostgreSQL/testcontainer infrastructure.

**Out of scope:**
- Testcontainer PostgreSQL infrastructure and CI branch-protection
  changes.
- Any compose-loop integration or user-visible route behavior from
  later phases.
- Frontend work.

**Exit gates:**
- The primitive cannot persist an inconsistent assistant/tool
  transcript.
- Generic SQLAlchemy and filesystem persistence failures preserve the
  primary plugin failure, emit allowed telemetry/audit fields only, and
  do not leak SQL params, row payloads, raw exception text, or secrets.
- Cancellation tests prove the documented retry/idempotency behavior.
- A follow-up review confirms B4, B5, B6, and W4 are resolved.

### Schedule C: PostgreSQL, CI, and Operational Proof

**Risk controlled:** dialect drift, Docker/testcontainer dependency
drift, deadlock/hang detection, CI gating, and final handoff accuracy.

**Scope:**
- Task 16 PostgreSQL/testcontainer lane after current metadata is made
  PostgreSQL-portable or the test scope is explicitly constrained.
- Task 17 latency sanity tests with bounded timeout behavior.
- Task 19 final spec/OQ amendments.
- Task 20 final CI and release-handback steps, adjusted to represent
  Schedule C as the final Phase 1 closure PR rather than a monolithic
  all-in-one PR.

**Out of scope:**
- New schema or primitive behavior except for fixes required by the
  PostgreSQL proof.
- Any speed-driven parallelization with Schedule A or B.

**Exit gates:**
- `initialize_session_schema(engine)` succeeds on PostgreSQL for the
  schema actually used by the testcontainer lane.
- The dependency check fails closed when `testcontainers.postgres` is
  missing.
- Concurrency tests assert all worker threads/futures terminate and
  fail with useful diagnostics on timeout.
- CI success aggregation includes the Docker/testcontainer lane.
- Final plan/spec anchors and blocker IDs are current enough for a
  future implementer to follow without relying on stale review JSON.

## File Structure

### Files to modify

- `src/elspeth/web/sessions/models.py` — add new columns and CHECK constraints to `chat_messages`; extend the role CHECK with the internal `"audit"` role for audit-only composer breadcrumbs that do not have a parent assistant row; add `provenance` column to `composition_states`; add `audit_access_log` table.
- `src/elspeth/web/sessions/service.py` — add the concrete sync primitive `persist_compose_turn`, the protocol-facing async dispatcher `persist_compose_turn_async`, `_acquire_session_advisory_lock`, `_reserve_sequence_range`, `_insert_chat_message`, `_insert_composition_state`, shared state-envelope helper, process-wide SQLite same-session serialization helpers, and stale-current-state checks; rewrite `add_message` (Task 14) to accept the required keyword-only `writer_principal` argument while preserving its pre-rev-4 behaviours (cross-session guard, `updated_at` write, `raw_content` persistence, `ChatMessageRecord` return). Task 10 keeps `save_composition_state` and `set_active_state` inline with provenance + lock additions, and refactors only `fork_session` to `_insert_composition_state`.
- `src/elspeth/web/sessions/protocol.py` — add `SessionServiceProtocol.persist_compose_turn_async` when Task 11 introduces the primitive (using `TYPE_CHECKING` imports for `_RedactedToolRow` / `_AuditOutcome` to avoid a runtime circular import with `_persist_payload.py`); the sync `SessionServiceImpl.persist_compose_turn` remains concrete-only and guarded against direct async-loop use. Update `SessionServiceProtocol.add_message` declaration at lines 258-266 to match the new signature (atomic with the service.py change in Task 14); extend `ChatMessageRole` / `ChatMessageRecord` to include the internal `"audit"` role and new tool linkage fields.
- `src/elspeth/web/sessions/routes.py` — update **all six** current production `service.add_message(...)` writers: `_persist_tool_invocations` (line ~703), `_persist_llm_calls` (line ~752), user message (line ~1599), assistant message send path (line ~1984), assistant message recompose path (line ~2431), and revert system message (line ~2641). Per-site mapping is in Task 14 §14.4. Always re-grep `service.add_message(` and `\.add_message(` in `routes.py` before editing; stale four-call-site inventories are explicitly wrong.
- `pyproject.toml` and `uv.lock` — add `testcontainers[postgres]`, register the `testcontainer` marker, and keep frozen CI sync reproducible.
- `.github/workflows/ci.yaml` — add an explicit Docker-enabled `pytest -m testcontainer tests/integration/web/ -v` lane or step; default non-Docker test jobs must continue deselecting `testcontainer`.
- `docs/superpowers/specs/2026-04-30-composer-progress-persistence-design.md` — amend or explicitly supersede stale Phase 1 snippets for role values, `_StatePayload`, advisory-lock SQL, `persist_compose_turn(raw_content, expected_current_state_id)`, the async dispatcher contract, `_AuditOutcome`, audit-access cascade, and session-test engine construction before implementation begins. This is a Phase 1 preflight deliverable, not a later observation-only follow-up.

### Files to create

- `src/elspeth/contracts/advisory_locks.py` — module-level constants reserving PostgreSQL advisory-lock classid namespaces. Phase 1 introduces `ELSPETH_SESSIONS_LOCK_CLASSID = 0x454C5350` (= 1,162,629,968, ASCII "ELSP" big-endian) for the sessions-DB session write lock; closes B3 from the Phase 1 plan-review synthesis (single-argument `pg_advisory_xact_lock` shares one cluster-wide namespace and could collide with any other application or ELSPETH subsystem on the same Postgres cluster). The constant is on-the-wire ABI: changing it across versions silently breaks mutual exclusion between old and new instances on the same cluster, and any future ELSPETH advisory lock MUST use a different value from this file's registry. ABI changes require an ADR with explicit migration plan.
- `src/elspeth/web/sessions/telemetry.py` — module-level `_SessionsTelemetry` dataclass holding the named OTel counters introduced in spec §1.4. Sessions owns the persistence counters; composer imports or app wiring may consume them, but `web/sessions/service.py` must not import from `web/composer`.
- `src/elspeth/web/sessions/_persist_payload.py` — small dataclass module: `_ToolOutcome`, `_RedactedToolRow`, `_StatePayload`, `_AuditOutcome`. Used by `persist_compose_turn` (Phase 1 introduces them; Phase 3 wires the compose loop to produce them). `_StatePayload` carries no `version` field per B1 from the Phase 1 plan-review synthesis — `_insert_composition_state` allocates the version under `_session_write_lock` to foreclose the dual-allocator race that would otherwise fabricate Tier-1 audit-integrity violations on contention loss.
- `tests/unit/web/sessions/test_chat_messages.py` — schema constraint tests (in-memory SQLite).
- `tests/unit/web/sessions/test_composition_states.py` — provenance CHECK + refactored inline inserts.
- `tests/unit/web/sessions/test_audit_access_log.py` — schema tests for the new table.
- `tests/unit/web/sessions/test_persist_compose_turn.py` — sync primitive unit tests.
- `tests/unit/web/composer/test_audit_failure_primacy.py` — primacy disposition tests.
- `tests/unit/web/sessions/test_telemetry.py` — counter wiring.
- `tests/integration/web/test_inv_audit_ahead_backward.py` — schema-level backward-direction post-condition.
- `tests/integration/web/test_compose_loop_concurrent_sessions.py` — CL-PP-11 (testcontainer PostgreSQL).
- `tests/integration/web/test_compose_loop_latency_sanity.py` — p95 sanity bound (in-memory SQLite).
- `docs/runbooks/staging-session-db-recreation.md` — document the session-DB archive/delete/restart procedure for pre-release schema changes (supersedes OQ-4's original "pre-deploy DELETE" framing, which did not match codebase reality). Use the existing runbook/docs convention instead of introducing `docs/operations/`.

### Files NOT touched in Phase 1

- `src/elspeth/web/composer/service.py` (compose loop). Phase 3.
- `src/elspeth/web/composer/redaction.py`. Phase 2.
- Anything under `src/elspeth/web/frontend/`. Phase 4.

---

## Task 0: Mark stale spec snippets superseded before implementation

**Why this is first.** Multiple Phase 1 corrections intentionally
supersede stale text in
`docs/superpowers/specs/2026-04-30-composer-progress-persistence-design.md`.
Leaving that spec as the governing handoff until Task 19 invites the
next implementer to copy the wrong lock SQL, stale `_StatePayload`
shape, or obsolete `persist_compose_turn` signature before the cleanup
task runs. The implementation plan is authoritative for Phase 1, but
the spec must say so mechanically at the start of the PR.

**Files:**
- Modify: `docs/superpowers/specs/2026-04-30-composer-progress-persistence-design.md`

- [ ] **Step 1: Add a supersession note near the spec's Phase 1 handoff text**

Add a dated note that points to this Phase 1 plan and names the stale
snippets it supersedes:

- `chat_messages.role` includes internal `"audit"`.
- `chat_messages.writer_principal` includes `"session_fork"`.
- `parent_assistant_id` is enforced by a composite same-session FK:
  `(parent_assistant_id, session_id) -> (chat_messages.id, chat_messages.session_id)`.
- `_StatePayload` has no `version`; `_insert_composition_state`
  allocates versions under `_session_write_lock`.
- PostgreSQL session write locks use
  `pg_advisory_xact_lock(ELSPETH_SESSIONS_LOCK_CLASSID, hashtext(session_id))`.
- `SessionServiceImpl.persist_compose_turn` accepts optional
  `raw_content` and `expected_current_state_id`, remains concrete-only,
  and is wrapped by the protocol-public async
  `SessionServiceProtocol.persist_compose_turn_async`.
- `_AuditOutcome` has only `assistant_id` and `unwind_audit_failed`;
  Tier-1 audit-write failures raise.
- Session tests use `create_session_engine(..., StaticPool)` plus
  `initialize_session_schema()`, never bare `metadata.create_all()`.

- [ ] **Step 2: Commit the supersession marker before code work**

```bash
git add docs/superpowers/specs/2026-04-30-composer-progress-persistence-design.md
git commit -m "docs(composer): mark Phase 1 persistence spec snippets superseded"
```

Task 19 still performs the final spec amendment after the implementation
has landed. Task 0 is the early warning label that prevents stale
handoff text from governing the code tasks.

---

## Task 1: Test infrastructure (conftests) + new `chat_messages` columns and CHECK constraints

**Why this task absorbs the conftest setup.** Every later task in
Phase 1 needs (a) an in-memory SQLite engine with FK enforcement
genuinely on for *every* pool checkout (not just one connection), and
(b) a way to insert sessions that satisfies all NOT NULL columns
(`user_id`, `auth_provider_type`, `title`, `updated_at`). The plan's
earlier draft used bare `create_engine("sqlite:///:memory:")` plus a
single `PRAGMA foreign_keys=ON` on one connection — that only enables
FK enforcement on that one connection, not on the pool's later
checkouts. Combined with `_run_sync`'s thread-pool dispatch, a worker
thread typically sees a *different* `:memory:` database with FK
enforcement off. Tests would silently pass for the wrong reason.
Putting the canonical fixture and session-insert helper into a
conftest before any other task uses them is the only consistent shape.

**Files:**
- Create: `tests/unit/web/conftest.py` (test helpers — applies to every Phase 1 unit test under `tests/unit/web/...`, including both `tests/unit/web/sessions/` and `tests/unit/web/composer/`. Hoisted to the parent package so the `_make_session` helper and `engine` fixture are visible to composer-suite tests — notably `tests/unit/web/composer/test_audit_failure_primacy.py` — without duplication. Synthesised review B5.)
- Create: `tests/integration/web/conftest.py` (test helpers — duplicates the `_make_session` helper for integration tests; the helper is one ~15-line function and parallel conftests is the standard pytest pattern for this codebase, see `tests/integration/checkpoint/conftest.py`)
- Modify: `src/elspeth/web/sessions/models.py`
- Test: `tests/unit/web/sessions/test_chat_messages.py` (create)

- [ ] **Step 1a: Create the unit-test conftest**

Create `tests/unit/web/conftest.py`:

```python
"""Shared fixtures and helpers for Phase 1 web unit tests.

Hoisted from ``tests/unit/web/sessions/conftest.py`` to the parent
``tests/unit/web/`` package so both the sessions suite
(``tests/unit/web/sessions/test_*.py``) and the composer suite
(``tests/unit/web/composer/test_audit_failure_primacy.py``) see the
same fixtures. pytest auto-loads parent-directory conftests for any
test under that directory tree, so no per-subdirectory shim is needed.

Provides:
- ``engine`` — an in-memory SQLite engine with FK enforcement applied
  via ``create_session_engine``'s connect-event listener (so EVERY
  pool checkout enforces FKs, not just the first), backed by
  ``StaticPool`` so worker threads dispatched via ``_run_sync`` see
  the same in-memory database. Schema is bootstrapped via
  ``initialize_session_schema`` (the same path production uses).
- ``_make_session`` — non-fixture helper that inserts a row into
  ``sessions_table`` with every NOT NULL column populated. Test code
  imports it explicitly via the absolute path
  ``from tests.unit.web.conftest import _make_session`` (matches the
  codebase convention for cross-package shared helpers — see
  ``tests/fixtures/`` and ``tests/helpers/`` import sites).

Why these live in conftest rather than each test file inlining them:
the four NOT NULL columns on ``sessions_table`` (``user_id``,
``auth_provider_type``, ``title``, ``updated_at``) and the
``StaticPool`` requirement are easy to forget. Centralising both
makes the fixture banned-pattern violations (plain ``create_engine``;
minimum-columns inserts) literally absent from the test code.
"""
from __future__ import annotations

from datetime import UTC, datetime

import pytest
from sqlalchemy import Connection, insert
from sqlalchemy.pool import StaticPool

from elspeth.web.sessions import models
from elspeth.web.sessions.engine import create_session_engine
from elspeth.web.sessions.schema import initialize_session_schema


@pytest.fixture
def engine():
    eng = create_session_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    initialize_session_schema(eng)
    return eng


def _make_session(
    conn: Connection,
    *,
    session_id: str,
    user_id: str = "test_user",
    auth_provider_type: str = "local",
    title: str = "test session",
    created_at: datetime | None = None,
    updated_at: datetime | None = None,
) -> None:
    """Insert a session row with every NOT NULL column populated.

    The defaults are sufficient for tests that do not care about
    user/auth-provider fields. Tests that exercise auth-scoped
    behaviour should pass explicit ``user_id`` and
    ``auth_provider_type`` values.
    """
    now = created_at or datetime.now(UTC)
    conn.execute(
        insert(models.sessions_table).values(
            id=session_id,
            user_id=user_id,
            auth_provider_type=auth_provider_type,
            title=title,
            created_at=now,
            updated_at=updated_at or now,
        )
    )
```

- [ ] **Step 1b: Create the integration-test conftest**

Create `tests/integration/web/conftest.py`:

```python
"""Shared helpers for Phase 1 web integration tests.

Duplicates ``_make_session`` from
``tests/unit/web/conftest.py``; if either copy changes the
other must be updated to match. Engine fixtures are NOT shared because
some integration tests use SQLite (``test_inv_audit_ahead_backward``,
``test_compose_loop_latency_sanity``) while CL-PP-11 uses a
testcontainer Postgres — each test owns its engine fixture and calls
``_make_session`` against whatever connection it has.
"""
from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import Connection, insert

from elspeth.web.sessions import models


def _make_session(
    conn: Connection,
    *,
    session_id: str,
    user_id: str = "test_user",
    auth_provider_type: str = "local",
    title: str = "test session",
    created_at: datetime | None = None,
    updated_at: datetime | None = None,
) -> None:
    """Insert a session row with every NOT NULL column populated."""
    now = created_at or datetime.now(UTC)
    conn.execute(
        insert(models.sessions_table).values(
            id=session_id,
            user_id=user_id,
            auth_provider_type=auth_provider_type,
            title=title,
            created_at=now,
            updated_at=updated_at or now,
        )
    )
```

- [ ] **Step 1c: Write the failing schema test**

Create `tests/unit/web/sessions/test_chat_messages.py`:

```python
"""Schema tests for the rev-4 chat_messages columns and CHECK constraints.

These tests run against an in-memory SQLite database to exercise the actual
database engine, not just SQLAlchemy metadata declarations. Schema-only
introspection would pass against any declared schema regardless of whether
the database enforces the declarations (closes spec QA F-8).
"""
from __future__ import annotations

from datetime import UTC, datetime

import pytest
from sqlalchemy import insert
from sqlalchemy.exc import IntegrityError

from elspeth.web.sessions import models

from tests.unit.web.conftest import _make_session


def test_tool_call_id_column_exists(engine):
    cols = {c.name for c in models.chat_messages_table.columns}
    assert "tool_call_id" in cols
    assert "parent_assistant_id" in cols
    assert "sequence_no" in cols
    assert "writer_principal" in cols


def test_role_tool_requires_tool_call_id(engine):
    with engine.begin() as conn:
        _make_session(conn, session_id="s1")
        with pytest.raises(IntegrityError, match="ck_chat_messages_tool_call_id_role"):
            conn.execute(insert(models.chat_messages_table).values(
                id="m1", session_id="s1", role="tool",
                content="{}", sequence_no=1, writer_principal="compose_loop",
                created_at=datetime(2026, 4, 30, tzinfo=UTC),
                tool_call_id=None,           # rejected by CHECK
                parent_assistant_id=None,
            ))


def test_role_assistant_rejects_tool_call_id(engine):
    with engine.begin() as conn:
        _make_session(conn, session_id="s1")
        with pytest.raises(IntegrityError, match="ck_chat_messages_tool_call_id_role"):
            conn.execute(insert(models.chat_messages_table).values(
                id="m1", session_id="s1", role="assistant",
                content="hi", sequence_no=1, writer_principal="compose_loop",
                created_at=datetime(2026, 4, 30, tzinfo=UTC),
                tool_call_id="should_not_be_set",
            ))


def test_writer_principal_check_rejects_unknown(engine):
    with engine.begin() as conn:
        _make_session(conn, session_id="s1")
        with pytest.raises(IntegrityError, match="ck_chat_messages_writer_principal"):
            conn.execute(insert(models.chat_messages_table).values(
                id="m1", session_id="s1", role="user",
                content="hi", sequence_no=1,
                writer_principal="rogue_writer",  # not in CHECK enum
                created_at=datetime(2026, 4, 30, tzinfo=UTC),
            ))


def test_audit_role_allows_unparented_internal_breadcrumb(engine):
    """Internal composer audit breadcrumbs that have no assistant parent
    are stored as role='audit', not role='tool'. This is the
    schema-compatible shape for existing _persist_llm_calls and
    failure-path _persist_tool_invocations rows until Phase 3 wires
    successful tool responses through persist_compose_turn."""
    with engine.begin() as conn:
        _make_session(conn, session_id="s1")
        conn.execute(insert(models.chat_messages_table).values(
            id="audit1",
            session_id="s1",
            role="audit",
            content='{"_kind": "llm_call_audit"}',
            sequence_no=1,
            writer_principal="compose_loop",
            tool_call_id=None,
            parent_assistant_id=None,
            created_at=datetime(2026, 4, 30, tzinfo=UTC),
        ))


# The ``ck_chat_messages_parent_role`` CHECK is a BICONDITIONAL on
# ``(role = 'tool') = (parent_assistant_id IS NOT NULL)``. The four
# tests below exercise all four cells of the truth table — closing
# synthesised review finding M7 / Q-F-05. Without these, a
# constraint that fired on one direction but not the other would
# pass the broader column-existence and tool-id tests above.


def test_parent_role_tool_with_parent_id_set_is_accepted(engine):
    """role='tool' AND parent_assistant_id IS NOT NULL — biconditional
    holds in the affirmative; insert succeeds. (Tool rows must also
    have tool_call_id and a parent assistant message in the session;
    the test sets both up.)"""
    with engine.begin() as conn:
        _make_session(conn, session_id="s1")
        # Insert the parent assistant message first so the FK on
        # parent_assistant_id can resolve.
        conn.execute(insert(models.chat_messages_table).values(
            id="a1", session_id="s1", role="assistant",
            content="hi", sequence_no=1, writer_principal="compose_loop",
            created_at=datetime(2026, 4, 30, tzinfo=UTC),
        ))
        conn.execute(insert(models.chat_messages_table).values(
            id="t1", session_id="s1", role="tool",
            content="{}", sequence_no=2, writer_principal="compose_loop",
            tool_call_id="tc_1", parent_assistant_id="a1",
            created_at=datetime(2026, 4, 30, tzinfo=UTC),
        ))


def test_parent_role_tool_without_parent_id_rejected(engine):
    """role='tool' AND parent_assistant_id IS NULL — biconditional
    violated; CHECK rejects. tool_call_id is set so this test
    isolates the parent_role CHECK from the tool_call_id_role CHECK."""
    with engine.begin() as conn:
        _make_session(conn, session_id="s1")
        with pytest.raises(IntegrityError, match="ck_chat_messages_parent_role"):
            conn.execute(insert(models.chat_messages_table).values(
                id="t1", session_id="s1", role="tool",
                content="{}", sequence_no=1, writer_principal="compose_loop",
                tool_call_id="tc_1",
                parent_assistant_id=None,    # rejected by parent_role CHECK
                created_at=datetime(2026, 4, 30, tzinfo=UTC),
            ))


def test_parent_role_non_tool_with_parent_id_rejected(engine):
    """role='user' AND parent_assistant_id IS NOT NULL — biconditional
    violated; CHECK rejects. tool_call_id is None (correct for user)
    so this test isolates the parent_role CHECK."""
    with engine.begin() as conn:
        _make_session(conn, session_id="s1")
        # Insert a real assistant first so the FK target exists; the
        # CHECK fires before the FK in SQLite, but we set up the FK
        # target either way to keep the failure unambiguous.
        conn.execute(insert(models.chat_messages_table).values(
            id="a1", session_id="s1", role="assistant",
            content="hi", sequence_no=1, writer_principal="compose_loop",
            created_at=datetime(2026, 4, 30, tzinfo=UTC),
        ))
        with pytest.raises(IntegrityError, match="ck_chat_messages_parent_role"):
            conn.execute(insert(models.chat_messages_table).values(
                id="u1", session_id="s1", role="user",
                content="hello", sequence_no=2,
                writer_principal="route_user_message",
                tool_call_id=None,
                parent_assistant_id="a1",   # rejected: non-tool roles MUST have NULL parent
                created_at=datetime(2026, 4, 30, tzinfo=UTC),
            ))


def test_parent_role_non_tool_without_parent_id_is_accepted(engine):
    """role='user' AND parent_assistant_id IS NULL — biconditional
    holds in the negative; insert succeeds. The fourth and final
    cell of the truth table."""
    with engine.begin() as conn:
        _make_session(conn, session_id="s1")
        conn.execute(insert(models.chat_messages_table).values(
            id="u1", session_id="s1", role="user",
            content="hello", sequence_no=1,
            writer_principal="route_user_message",
            tool_call_id=None,
            parent_assistant_id=None,
            created_at=datetime(2026, 4, 30, tzinfo=UTC),
        ))


def test_session_sequence_no_unique(engine):
    with engine.begin() as conn:
        _make_session(conn, session_id="s1")
        conn.execute(insert(models.chat_messages_table).values(
            id="m1", session_id="s1", role="user",
            content="hi", sequence_no=1, writer_principal="route_user_message",
            created_at=datetime(2026, 4, 30, tzinfo=UTC),
        ))
        # SQLite reports unique-index violations with the column names
        # rather than the index name, so anchor the match on the
        # column-list rather than ``ix_chat_messages_session_sequence``.
        with pytest.raises(IntegrityError, match=r"UNIQUE.*chat_messages.*sequence_no"):
            conn.execute(insert(models.chat_messages_table).values(
                id="m2", session_id="s1", role="assistant",
                content="hi", sequence_no=1,  # duplicate
                writer_principal="compose_loop",
                created_at=datetime(2026, 4, 30, tzinfo=UTC),
            ))


def test_delete_assistant_row_cascades_to_tool_rows(engine):
    """Spec §8.1: ``Assert ON DELETE CASCADE from session removes all rows;
    from assistant row removes child tool rows (orphan prevention).``

    This is a Tier-1 referential-integrity invariant on the audit DB. The
    column-existence and CHECK-constraint tests above only inspect schema
    metadata; a future migration that drops ``ondelete='CASCADE'`` from
    the ``parent_assistant_id`` FK would pass those tests silently and
    leave orphaned tool rows referencing a deleted assistant. This test
    binds the invariant to behaviour: insert one assistant + two child
    tool rows, DELETE the assistant, and assert the children are gone.

    FK enforcement on SQLite requires ``PRAGMA foreign_keys=ON`` for
    every connection in the pool. The ``engine`` fixture builds the
    engine via ``create_session_engine`` (see
    ``src/elspeth/web/sessions/engine.py``), which registers a
    ``connect`` event listener that issues the PRAGMA on every DBAPI
    connection and asserts on first connect that the PRAGMA took
    effect. Combined with the conftest's ``StaticPool`` (so worker
    threads see the same in-memory database), CASCADE genuinely fires
    here and cannot silently no-op the way bare ``create_engine``
    would.
    """
    with engine.begin() as conn:
        _make_session(conn, session_id="s1")
        # Parent assistant row (no tool_call_id, no parent_assistant_id —
        # consistent with the biconditional CHECK constraints).
        conn.execute(insert(models.chat_messages_table).values(
            id="a1", session_id="s1", role="assistant",
            content="hi", sequence_no=1, writer_principal="compose_loop",
            created_at=datetime(2026, 4, 30, tzinfo=UTC),
        ))
        # Two child tool rows pointing back at a1; distinct tool_call_ids
        # so the partial unique index (Task 2) is not implicated here.
        conn.execute(insert(models.chat_messages_table).values(
            id="t1", session_id="s1", role="tool",
            content="{}", sequence_no=2, writer_principal="compose_loop",
            tool_call_id="tc_1", parent_assistant_id="a1",
            created_at=datetime(2026, 4, 30, tzinfo=UTC),
        ))
        conn.execute(insert(models.chat_messages_table).values(
            id="t2", session_id="s1", role="tool",
            content="{}", sequence_no=3, writer_principal="compose_loop",
            tool_call_id="tc_2", parent_assistant_id="a1",
            created_at=datetime(2026, 4, 30, tzinfo=UTC),
        ))

        # Sanity: all three rows are present before the delete.
        pre_rows = conn.execute(
            select(models.chat_messages_table.c.id)
            .where(models.chat_messages_table.c.session_id == "s1")
        ).scalars().all()
        assert sorted(pre_rows) == ["a1", "t1", "t2"]

        # Delete the assistant row. ON DELETE CASCADE on
        # parent_assistant_id MUST remove t1 and t2 in the same statement.
        conn.execute(
            delete(models.chat_messages_table)
            .where(models.chat_messages_table.c.id == "a1")
        )

        post_rows = conn.execute(
            select(models.chat_messages_table.c.id)
            .where(models.chat_messages_table.c.session_id == "s1")
        ).scalars().all()
        assert post_rows == [], (
            "Expected ON DELETE CASCADE on parent_assistant_id to remove "
            "child tool rows when the assistant row is deleted; found "
            f"orphaned rows: {post_rows!r}. If this test fails the audit "
            "DB has lost orphan-prevention — investigate the FK definition "
            "in models.chat_messages_table and the engine's foreign_keys "
            "PRAGMA before re-enabling."
        )


def test_tool_row_rejects_cross_session_parent_assistant(engine):
    """A tool row's parent assistant must belong to the same session.

    The old one-column FK on ``parent_assistant_id`` only proved that
    the assistant row existed somewhere in ``chat_messages``. It allowed
    a tool row in session B to point at an assistant row in session A,
    which would splice two transcripts together in the audit DB. The
    composite FK
    ``(parent_assistant_id, session_id) -> (chat_messages.id, chat_messages.session_id)``
    is the mechanical invariant.
    """
    with engine.begin() as conn:
        _make_session(conn, session_id="s_parent")
        _make_session(conn, session_id="s_child")
        conn.execute(insert(models.chat_messages_table).values(
            id="a_parent", session_id="s_parent", role="assistant",
            content="hi", sequence_no=1, writer_principal="compose_loop",
            created_at=datetime(2026, 4, 30, tzinfo=UTC),
        ))
        with pytest.raises(
            IntegrityError,
            match=r"FOREIGN KEY|fk_chat_messages_parent_assistant_session",
        ):
            conn.execute(insert(models.chat_messages_table).values(
                id="t_child", session_id="s_child", role="tool",
                content="{}", sequence_no=1, writer_principal="compose_loop",
                tool_call_id="tc_cross", parent_assistant_id="a_parent",
                created_at=datetime(2026, 4, 30, tzinfo=UTC),
            ))
```

- [ ] **Step 2: Run test to verify it fails**

```bash
.venv/bin/python -m pytest tests/unit/web/sessions/test_chat_messages.py -v
```
Expected: FAIL — columns do not exist yet. The failing tests include
`test_tool_call_id_column_exists`, the four-cell biconditional truth-table
tests for `ck_chat_messages_parent_role`, the `tool_call_id_role` CHECK
tests, the `writer_principal` CHECK test, the unique
`(session_id, sequence_no)` index test, and the new
`test_delete_assistant_row_cascades_to_tool_rows` cascade-behaviour
test plus `test_tool_row_rejects_cross_session_parent_assistant` (B7
and B8 — closes the Tier-1 orphan-prevention and same-session parent
invariants in spec §8.1).

- [ ] **Step 3: Add columns and CHECK constraints in models.py**

Modify `src/elspeth/web/sessions/models.py`'s `chat_messages_table` definition to match spec §4.1.1:

```python
chat_messages_table = Table(
    "chat_messages",
    metadata,
    Column("id", String, primary_key=True),
    Column(
        "session_id",
        String,
        ForeignKey("sessions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    ),
    Column("role", String, nullable=False),
    Column("content", Text, nullable=False),
    Column("raw_content", Text, nullable=True),
    Column("tool_calls", JSON, nullable=True),
    Column("tool_call_id", String, nullable=True),
    Column("sequence_no", Integer, nullable=False),
    Column("writer_principal", String, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("composition_state_id", String, nullable=True),
    Column("parent_assistant_id", String, nullable=True),
    ForeignKeyConstraint(
        ["composition_state_id", "session_id"],
        ["composition_states.id", "composition_states.session_id"],
        name="fk_chat_messages_composition_state_session",
    ),
    ForeignKeyConstraint(
        ["parent_assistant_id", "session_id"],
        ["chat_messages.id", "chat_messages.session_id"],
        name="fk_chat_messages_parent_assistant_session",
        ondelete="CASCADE",
    ),
    UniqueConstraint(
        "id",
        "session_id",
        name="uq_chat_messages_id_session",
    ),
    CheckConstraint(
        "role IN ('user', 'assistant', 'system', 'tool', 'audit')",
        name="ck_chat_messages_role",
    ),
    CheckConstraint(
        "(role = 'tool') = (tool_call_id IS NOT NULL)",
        name="ck_chat_messages_tool_call_id_role",
    ),
    CheckConstraint(
        "(role = 'tool') = (parent_assistant_id IS NOT NULL)",
        name="ck_chat_messages_parent_role",
    ),
    CheckConstraint(
        "writer_principal IN ('compose_loop', 'route_user_message', "
        "'route_system_message', 'admin_tool', 'session_fork')",
        name="ck_chat_messages_writer_principal",
    ),
    Index(
        "ix_chat_messages_session_sequence",
        "session_id",
        "sequence_no",
        unique=True,
    ),
    Index(
        "ix_chat_messages_session_tool_call_id",
        "session_id",
        "tool_call_id",
    ),
)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
.venv/bin/python -m pytest tests/unit/web/sessions/test_chat_messages.py -v
```
Expected: PASS for every test in the file — column-existence,
`tool_call_id_role` CHECK (both directions), `parent_role` biconditional
truth table (all four cells), `writer_principal` CHECK,
`(session_id, sequence_no)` unique index, the composite
`parent_assistant_id` same-session FK, and
`test_delete_assistant_row_cascades_to_tool_rows` (the new B7 cascade
behaviour test). If the cascade test fails after the schema change,
inspect (1) `fk_chat_messages_parent_assistant_session` references
`(chat_messages.id, chat_messages.session_id)` and includes
`ondelete="CASCADE"`, (2) `uq_chat_messages_id_session` exists as the
composite FK target, and (3) the engine's `PRAGMA foreign_keys=ON`
listener fired — `create_session_engine` asserts this on first
connect, so a failure here implies the test is using a different
engine factory.

- [ ] **Step 5: Commit**

```bash
git add src/elspeth/web/sessions/models.py tests/unit/web/sessions/test_chat_messages.py
git commit -m "feat(sessions): add chat_messages tool_call_id/sequence_no/writer_principal columns and CHECK constraints (composer-progress-persistence phase 1)"
```

---

## Task 2: Add partial unique index on `(session_id, tool_call_id) WHERE role='tool'`

**Files:**
- Modify: `src/elspeth/web/sessions/models.py`
- Test: `tests/unit/web/sessions/test_chat_messages.py` (extend)

- [ ] **Step 1: Write the failing test**

Add to `tests/unit/web/sessions/test_chat_messages.py`:

```python
def test_tool_call_id_unique_within_session(engine):
    with engine.begin() as conn:
        _make_session(conn, session_id="s1")
        # Insert assistant first to satisfy parent_assistant_id FK
        conn.execute(insert(models.chat_messages_table).values(
            id="a1", session_id="s1", role="assistant",
            content="", sequence_no=1, writer_principal="compose_loop",
            created_at=datetime(2026, 4, 30, tzinfo=UTC),
        ))
        conn.execute(insert(models.chat_messages_table).values(
            id="t1", session_id="s1", role="tool",
            content="{}", sequence_no=2, writer_principal="compose_loop",
            tool_call_id="dup_id", parent_assistant_id="a1",
            created_at=datetime(2026, 4, 30, tzinfo=UTC),
        ))
        # SQLite reports the column list rather than the partial-index
        # name; PostgreSQL may include the named index. Match both so
        # the regression fails for the intended uniqueness contract.
        with pytest.raises(
            IntegrityError,
            match=(
                r"(UNIQUE.*chat_messages.*session_id.*tool_call_id"
                r"|uq_chat_messages_tool_call_id)"
            ),
        ):
            conn.execute(insert(models.chat_messages_table).values(
                id="t2", session_id="s1", role="tool",
                content="{}", sequence_no=3, writer_principal="compose_loop",
                tool_call_id="dup_id",  # duplicate
                parent_assistant_id="a1",
                created_at=datetime(2026, 4, 30, tzinfo=UTC),
            ))


def test_tool_call_id_unique_only_within_role_tool(engine):
    """The partial unique index excludes role!='tool' rows so user/assistant
    rows with NULL tool_call_id do not all collide on NULL."""
    with engine.begin() as conn:
        _make_session(conn, session_id="s1")
        conn.execute(insert(models.chat_messages_table).values(
            id="u1", session_id="s1", role="user",
            content="hi", sequence_no=1, writer_principal="route_user_message",
            created_at=datetime(2026, 4, 30, tzinfo=UTC),
        ))
        # Should not collide despite both having NULL tool_call_id.
        conn.execute(insert(models.chat_messages_table).values(
            id="a1", session_id="s1", role="assistant",
            content="hi", sequence_no=2, writer_principal="compose_loop",
            created_at=datetime(2026, 4, 30, tzinfo=UTC),
        ))
```

- [ ] **Step 2: Run test to verify it fails**

```bash
.venv/bin/python -m pytest tests/unit/web/sessions/test_chat_messages.py::test_tool_call_id_unique_within_session -v
```
Expected: FAIL — partial unique index does not exist.

- [ ] **Step 3: Add the partial unique index**

In `src/elspeth/web/sessions/models.py`, after the table definition,
declare the partial unique index using SQLAlchemy's native
``Index(..., sqlite_where=..., postgresql_where=...)`` pattern. This
matches the project-standard pattern already used at
``models.py:160-165`` (`uq_runs_one_active_per_session`) and avoids
the raw-DDL event-listener mechanism. Closes synthesised review
finding A-F13 / L5.

```python
# Partial unique constraint: tool_call_id must be unique within
# (session_id, tool_role) scope. The same predicate is supplied to
# both ``sqlite_where`` (SQLite 3.8.0+) and ``postgresql_where``
# (PostgreSQL ≥ 9.5) so the index is equivalent across dialects.
Index(
    "uq_chat_messages_tool_call_id",
    chat_messages_table.c.session_id,
    chat_messages_table.c.tool_call_id,
    unique=True,
    sqlite_where=chat_messages_table.c.role == "tool",
    postgresql_where=chat_messages_table.c.role == "tool",
)
```

(`Index` is already imported at the top of `models.py`. No `DDL`
or `event` imports are needed.)

- [ ] **Step 4: Run tests to verify pass**

```bash
.venv/bin/python -m pytest tests/unit/web/sessions/test_chat_messages.py -v
```
Expected: PASS for the two new tests; previously-passing tests still pass.

- [ ] **Step 5: Commit**

```bash
git add src/elspeth/web/sessions/models.py tests/unit/web/sessions/test_chat_messages.py
git commit -m "feat(sessions): add partial unique index uq_chat_messages_tool_call_id (composer-progress-persistence phase 1)"
```

---

## Task 3: Add `provenance` column to `composition_states`

**Files:**
- Modify: `src/elspeth/web/sessions/models.py`
- Test: `tests/unit/web/sessions/test_composition_states.py` (create)

- [ ] **Step 1: Write the failing test**

Create `tests/unit/web/sessions/test_composition_states.py`:

```python
"""Schema tests for the rev-4 composition_states.provenance column.

Uses the shared ``engine`` fixture and ``_make_session`` helper from
``tests/unit/web/conftest.py``.
"""
from __future__ import annotations

from datetime import UTC, datetime

import pytest
from sqlalchemy import insert
from sqlalchemy.exc import IntegrityError

from elspeth.web.sessions import models

from tests.unit.web.conftest import _make_session


def test_provenance_column_exists():
    cols = {c.name for c in models.composition_states_table.columns}
    assert "provenance" in cols


def test_provenance_check_accepts_known_values(engine):
    """Each provenance value listed in the spec §4.1.2 enum must insert
    cleanly. The test exercises the per-row CHECK in isolation; it does
    not assert anything about the semantic correctness of each value
    (that is the spec's concern, not the schema's)."""
    now = datetime.now(UTC)
    for provenance in (
        "tool_call",
        "convergence_persist",
        "plugin_crash_persist",
        "preflight_persist",
        "session_seed",
        "session_fork",
    ):
        with engine.begin() as conn:
            _make_session(conn, session_id=f"s_{provenance}")
            conn.execute(insert(models.composition_states_table).values(
                id=f"cs_{provenance}",
                session_id=f"s_{provenance}",
                version=1,
                provenance=provenance,
                created_at=now,
                # is_valid has a Python-side default of False on the
                # column declaration, so it does not need to be passed
                # explicitly. All JSON content columns are nullable.
            ))


def test_provenance_check_rejects_unknown_value(engine):
    with engine.begin() as conn:
        _make_session(conn, session_id="s1")
        with pytest.raises(IntegrityError, match="ck_composition_states_provenance"):
            conn.execute(insert(models.composition_states_table).values(
                id="cs_x", session_id="s1",
                version=1, provenance="rogue_value",
                created_at=datetime.now(UTC),
            ))


def test_provenance_not_null(engine):
    with engine.begin() as conn:
        _make_session(conn, session_id="s1")
        # SQLite reports NOT NULL violations as
        # ``NOT NULL constraint failed: composition_states.provenance``,
        # so anchor on the column reference rather than a constraint name.
        with pytest.raises(IntegrityError, match=r"NOT NULL.*composition_states.provenance"):
            conn.execute(insert(models.composition_states_table).values(
                id="cs_x", session_id="s1",
                version=1, provenance=None,
                created_at=datetime.now(UTC),
            ))
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv/bin/python -m pytest tests/unit/web/sessions/test_composition_states.py -v
```
Expected: FAIL — `provenance` column missing.

- [ ] **Step 3: Add the column to models.py**

In `src/elspeth/web/sessions/models.py`'s `composition_states_table`:

```python
# Add inside the existing Table(...) call, alongside other columns:
Column("provenance", String, nullable=False),
# And at the constraints section:
CheckConstraint(
    "provenance IN ('tool_call', 'convergence_persist', "
    "'plugin_crash_persist', 'preflight_persist', "
    "'session_seed', 'session_fork')",
    name="ck_composition_states_provenance",
),
```

> **Spec amendment proposal — §4.1.2 enum and `session_seed` semantics.**
>
> The earlier plan draft used five values
> (`tool_call`, `convergence_persist`, `plugin_crash_persist`,
> `preflight_persist`, `session_seed`). The synthesised review panel
> (finding H9) flagged that the existing inline `composition_states`
> inserts in `service.py` have richer semantics than the spec's
> narrow `session_seed` definition covers:
>
> - `service.py:~403` (`save_composition_state`): general route-level
>   state save (NEW state, no lineage).
> - `service.py:~834` (`set_active_state`): state revert/pin within a
>   session (NEW state, derived from a prior state in the same
>   session).
> - `service.py:~1191` (`fork_session`): state COPIED to a NEW
>   session at fork time (cross-session derivation).
>
> Only the third site is a literal session fork. Adding a single new
> value `session_fork` for that case is the minimum-friction honest
> fix. The first two sites (general route saves and same-session
> state pins) reuse the existing `session_seed` value, which this
> task broadens to mean "any state row not produced by the compose
> loop" — a documented semantic widening, not a silent reinterpretation.
>
> Spec §4.1.2 needs the corresponding amendment:
>   - Add `session_fork` to the enum.
>   - Broaden the `session_seed` definition from "initial state row
>     written when a session is created with seed configuration" to
>     "any state row written outside the compose loop's tool-call
>     path: initial seed configuration, route-level state saves
>     (e.g. `save_composition_state`), and within-session state
>     reverts (e.g. `set_active_state`)."
>
> File the amendment in the same PR as Phase 1 so the spec and code
> ship together. Site-by-site mapping is enumerated in Task 10.

- [ ] **Step 4: Run tests to verify they pass**

```bash
.venv/bin/python -m pytest tests/unit/web/sessions/test_composition_states.py -v
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/elspeth/web/sessions/models.py tests/unit/web/sessions/test_composition_states.py
git commit -m "feat(sessions): add composition_states.provenance discriminator (composer-progress-persistence phase 1)"
```

---

## Task 4: Add `audit_access_log` table

**Files:**
- Modify: `src/elspeth/web/sessions/models.py`
- Test: `tests/unit/web/sessions/test_audit_access_log.py` (create)

- [ ] **Step 1: Write the failing test**

Create `tests/unit/web/sessions/test_audit_access_log.py`:

```python
"""Schema tests for the rev-4 audit_access_log table (spec §6.3).

Uses the shared ``engine`` fixture and ``_make_session`` helper from
``tests/unit/web/conftest.py``.
"""
from __future__ import annotations

from datetime import UTC, datetime

import pytest
from sqlalchemy import delete, insert, select
from sqlalchemy.exc import IntegrityError

from elspeth.web.sessions import models

from tests.unit.web.conftest import _make_session


def test_table_exists():
    assert "audit_access_log" in models.metadata.tables


def test_writer_principal_check(engine):
    now = datetime.now(UTC)
    with engine.begin() as conn:
        _make_session(conn, session_id="s1")
        # Accepted values
        for principal in ("audit_grade_view", "admin_tool"):
            conn.execute(insert(models.audit_access_log_table).values(
                id=f"al_{principal}",
                timestamp=now,
                session_id="s1",
                requesting_principal="user_42",
                request_path="/api/sessions/s1/messages",
                query_args={"include_tool_rows": True},
                ip_address="10.0.0.1",
                writer_principal=principal,
            ))
        # Rejected
        with pytest.raises(IntegrityError, match="ck_audit_access_log_writer_principal"):
            conn.execute(insert(models.audit_access_log_table).values(
                id="al_rogue",
                timestamp=now,
                session_id="s1",
                requesting_principal="user_42",
                request_path="/api/sessions/s1/messages",
                query_args={},
                writer_principal="rogue_view",
            ))


def test_session_delete_cascades_audit_access_log(engine):
    """Archive/delete lifecycle guard.

    ``archive_session`` ultimately deletes the parent session row after
    deleting the child tables it already knows about. Phase 3 writes
    ``audit_access_log`` rows, so the FK must cascade or archived sessions
    that have been viewed with ``include_tool_rows`` will fail deletion.
    """
    now = datetime.now(UTC)
    with engine.begin() as conn:
        _make_session(conn, session_id="s_archive")
        conn.execute(insert(models.audit_access_log_table).values(
            id="log1",
            timestamp=now,
            session_id="s_archive",
            requesting_principal="alice",
            request_path="/api/sessions/s_archive/messages",
            query_args={"include_tool_rows": True},
            ip_address=None,
            writer_principal="audit_grade_view",
        ))
        conn.execute(
            delete(models.sessions_table)
            .where(models.sessions_table.c.id == "s_archive")
        )
        remaining = conn.execute(
            select(models.audit_access_log_table.c.id)
            .where(models.audit_access_log_table.c.session_id == "s_archive")
        ).fetchall()
        assert remaining == []
```

- [ ] **Step 2: Run test to verify it fails**

```bash
.venv/bin/python -m pytest tests/unit/web/sessions/test_audit_access_log.py -v
```
Expected: FAIL — table does not exist.

- [ ] **Step 3: Add the table definition**

In `src/elspeth/web/sessions/models.py`, alongside the other table definitions:

```python
audit_access_log_table = Table(
    "audit_access_log",
    metadata,
    Column("id", String, primary_key=True),
    Column("timestamp", DateTime(timezone=True), nullable=False),
    Column(
        "session_id",
        String,
        ForeignKey("sessions.id", ondelete="CASCADE"),
        nullable=False,
    ),
    Column("requesting_principal", String, nullable=False),
    Column("request_path", String, nullable=False),
    Column("query_args", JSON, nullable=False),
    Column("ip_address", String, nullable=True),
    Column("writer_principal", String, nullable=False),
    CheckConstraint(
        "writer_principal IN ('audit_grade_view', 'admin_tool')",
        name="ck_audit_access_log_writer_principal",
    ),
    Index("ix_audit_access_log_session_timestamp", "session_id", "timestamp"),
)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
.venv/bin/python -m pytest tests/unit/web/sessions/test_audit_access_log.py -v
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/elspeth/web/sessions/models.py tests/unit/web/sessions/test_audit_access_log.py
git commit -m "feat(sessions): add audit_access_log table for audit-grade transcript view (composer-progress-persistence phase 1)"
```

---

## Task 5: Telemetry counters module + wire into `SessionServiceImpl` constructor

**Why this task includes the constructor wiring.** Tasks 7–13 reference
`self._telemetry` and `self._log` on `SessionServiceImpl`. The current
constructor (`service.py:86`) takes only `(engine, data_dir=None)` —
neither attribute exists. Extending the constructor independently of
the telemetry module would create a circular dependency (constructor
needs `_SessionsTelemetry` type; telemetry module exists only after this
task ships); deferring the extension to Task 7 would mean every
intervening task references attributes that don't yet exist. Bundling
both concerns in one atomic task is the only consistent shape.

The production wiring at `app.py:391` is updated in the same commit,
per CLAUDE.md no-legacy / single-commit-hard-cut policy: every caller
that constructs `SessionServiceImpl` is updated atomically with the
signature change.

**Current call-site reality.** The Phase 1 plan-review pass found 14
`SessionServiceImpl(` call sites across `src` and `tests`, including
`tests/unit/web/blobs/test_routes.py` outside the sessions test tree.
Task 5 is not complete until the implementer runs
`rg -n "SessionServiceImpl\\(" src tests -g '*.py'`, updates every
existing construction site, and pastes the before/after inventory into
the PR body. Narrowly updating `app.py` plus the new constructor test
false-greens this task.

**Files:**
- Create: `src/elspeth/web/sessions/telemetry.py`
- Create: `tests/unit/web/sessions/test_telemetry.py`
- Modify: `src/elspeth/web/sessions/service.py` (extend `__init__`)
- Modify: `src/elspeth/web/app.py` (update production wiring at line 391)
- Modify: every current `SessionServiceImpl(...)` construction site found by `rg -n "SessionServiceImpl\\(" src tests -g '*.py'`, including the existing sessions tests and `tests/unit/web/blobs/test_routes.py`.
- Test: `tests/unit/web/sessions/test_service_construction.py` (create — verifies the new constructor signature)

- [ ] **Step 1: Write the failing test**

Create `tests/unit/web/sessions/test_telemetry.py`:

```python
"""Tests for the named OTel counters introduced in spec §1.4 / §5.7.4."""
from __future__ import annotations

from elspeth.web.sessions.telemetry import (
    _FakeCounter,
    build_sessions_telemetry,
    observed_value,
)

# Spec §1.4 NFR table — production OTel metric strings for Phase 1.
# Verified end-to-end by ``test_production_meter_registers_named_metrics``
# below.
#
# Phase 1 ships ONLY the four audit-primacy counters that
# persist_compose_turn writes. The remaining four spec-§1.4 counters
# (``summarizer_errors_total``, ``unknown_response_key_total``,
# ``tool_call_cap_exceeded_total``, ``audit_grade_view_total``) are
# Phase 2/3 territory:
#
#   - summarizer_errors_total + unknown_response_key_total → Phase 2
#     (redaction primitives produce them).
#   - tool_call_cap_exceeded_total + audit_grade_view_total → Phase 3
#     (compose-loop cap and audit-grade view emit them).
#
# Adding them to Phase 1 would ship eight metric names of which four
# never increment — operationally indistinguishable from broken
# counters. Each Phase introduces its own counters when the code
# paths that emit them ship. Closes synthesised review finding M14
# / SA-4 (phase scope leak).
EXPECTED_METRIC_NAMES = {
    "composer.audit.tool_row_tier1_violation_total",
    "composer.audit.state_rolled_back_during_persist_total",
    "composer.audit.tool_row_persist_failed_during_unwind_total",
    "composer.audit.tool_row_integrity_violation_total",
}


def test_telemetry_field_names_match_spec_exactly():
    """Use ``set ==`` (not ``issubset``) so an accidental rename — say
    ``tier1_violation_total`` losing its ``tool_row_`` prefix — fails
    the test rather than passing under ``issubset``. Closes synthesised
    review finding L10 / Q-F-13."""
    telem = build_sessions_telemetry()
    expected_fields = {
        "tool_row_tier1_violation_total",
        "state_rolled_back_during_persist_total",
        "tool_row_persist_failed_during_unwind_total",
        "tool_row_integrity_violation_total",
    }
    actual = set(telem.__dataclass_fields__)
    assert actual == expected_fields, (
        f"field-name mismatch — added: {actual - expected_fields}; "
        f"removed: {expected_fields - actual}"
    )


def test_counter_increments_visible_via_observed_value_helper():
    """Test path: build_sessions_telemetry() with no meter returns
    fake counters; the ``observed_value`` helper extracts cumulative
    sum after type-narrowing to ``_FakeCounter``. Production code
    never reads ``observed_value`` — it only writes via ``add()`` —
    so the helper makes the test-only inspection explicit at the
    call site."""
    telem = build_sessions_telemetry()
    starting = observed_value(telem.tool_row_tier1_violation_total)
    telem.tool_row_tier1_violation_total.add(1)
    assert observed_value(telem.tool_row_tier1_violation_total) == starting + 1


def test_counter_records_attributes_dict():
    """Real OTel ``Counter.add`` accepts ``attributes`` as the second
    positional/keyword argument. Production code at composer/service.py
    and routes.py uses this for structured emission (e.g.,
    ``add(1, {"outcome": "failure"})``). The fake must mirror the
    signature so tests with attributed metrics do not raise
    ``TypeError`` against a fake-narrow ``add(amount)`` signature.
    Closes synthesised review finding H6."""
    telem = build_sessions_telemetry()
    telem.tool_row_tier1_violation_total.add(
        1, {"reason": "commit_failure", "session_id": "s_test"}
    )
    fake = telem.tool_row_tier1_violation_total
    assert isinstance(fake, _FakeCounter)
    assert fake.calls == [
        (1, {"reason": "commit_failure", "session_id": "s_test"}, None),
    ]


def test_production_meter_registers_named_metrics():
    """Closes synthesised review finding F-10 / L7. Verifies that the
    four Phase-1 ``meter.create_counter(...)`` strings in
    ``build_sessions_telemetry`` match spec §1.4 exactly. Without this
    test, a typo (e.g. ``tool_row_tier1_violations_total`` with a
    spurious ``s``) would pass the field-name check (which inspects
    Python attribute names, not OTel metric names) and silently break
    production observability."""

    class _RecordingMeter:
        """Captures the names passed to ``create_counter`` so the
        test can assert them as a set."""

        def __init__(self) -> None:
            self.registered: dict[str, _FakeCounter] = {}

        def create_counter(self, name: str) -> _FakeCounter:
            counter = _FakeCounter()
            self.registered[name] = counter
            return counter

    meter = _RecordingMeter()
    build_sessions_telemetry(meter=meter)
    assert set(meter.registered.keys()) == EXPECTED_METRIC_NAMES
```

- [ ] **Step 2: Run test to verify it fails**

```bash
.venv/bin/python -m pytest tests/unit/web/sessions/test_telemetry.py -v
```
Expected: FAIL — module does not exist.

- [ ] **Step 3: Implement the telemetry module**

Create `src/elspeth/web/sessions/telemetry.py`:

```python
"""OTel counters for SessionServiceImpl and the compose loop.

Production code uses the real OTel meter from
``opentelemetry.metrics.get_meter``; tests use
``build_sessions_telemetry()`` with no meter, which returns
``_FakeCounter`` instances. The ``_Counter`` and ``_Meter`` Protocols
match the OTel API exactly (``add(amount, attributes=None,
context=None)`` and ``create_counter(name, ...)``), so production
wiring type-checks without ``# type: ignore`` and the real meter
satisfies the structural contract.

The fake counter records every ``add`` call as ``(amount,
attributes, context)`` tuples. Tests inspect via the ``observed_value(counter)``
helper (cumulative sum) or directly through the ``calls`` attribute
after type-narrowing with ``isinstance(counter, _FakeCounter)``.
``observed_value`` is intentionally NOT on the ``_Counter`` Protocol
— production OTel counters do not expose observation, and adding it
would force a structural lie.

**Ownership-vs-metric namespace.** The container type is named
``_SessionsTelemetry`` and the module lives under
``web/sessions/telemetry.py`` because ``SessionServiceImpl`` owns
the persistence counters. The OTel metric strings remain
``composer.audit.*`` because operators consume them as part of the
composer-progress surface, but metric naming does not justify an
import from ``web/sessions/service.py`` up into ``web/composer``.
Composer code may import the sessions-owned container or receive it
from app wiring; sessions code must not import composer-owned modules.
Phase 2 (redaction counters) and Phase 3 (compose-loop and
audit-grade counters) may extend this container only if ownership
still belongs to the sessions persistence surface; otherwise those
phases add composer-owned telemetry separately.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol, TypeAlias

from opentelemetry.context import Context


_AttributeValue: TypeAlias = (
    str
    | bool
    | int
    | float
    | Sequence[str]
    | Sequence[bool]
    | Sequence[int]
    | Sequence[float]
)
_Attributes: TypeAlias = Mapping[str, _AttributeValue]


class _Counter(Protocol):
    """Subset of ``opentelemetry.metrics.Counter`` that production
    code uses.

    The real OTel signature is ``add(amount, attributes=None,
    context=None)``. Keep this Protocol as broad as the SDK surface that
    callers may legally use: ``amount`` may be ``int`` or ``float``;
    attributes may include every OTel scalar/sequence value type; and
    ``context`` is accepted even though Phase 1 callers omit it. This
    avoids a fake-narrow structural type that passes local tests but
    rejects a real Counter-compatible call shape.
    """

    def add(
        self,
        amount: int | float,
        attributes: _Attributes | None = None,
        context: Context | None = None,
    ) -> None: ...


class _Meter(Protocol):
    """Subset of ``opentelemetry.metrics.Meter`` that
    ``build_sessions_telemetry`` uses for production wiring."""

    def create_counter(self, name: str) -> _Counter: ...


class _FakeCounter:
    """Test-only counter that records every ``add`` call.

    Implements the ``_Counter`` Protocol structurally and adds the
    test inspection surface (``calls`` and the ``observed_value``
    helper). Production code MUST NOT depend on this class — it is
    re-exported only so the telemetry test module and the
    audit-failure-primacy tests can construct it directly.
    """

    def __init__(self) -> None:
        self.calls: list[
            tuple[int | float, dict[str, _AttributeValue] | None, Context | None]
        ] = []

    def add(
        self,
        amount: int | float,
        attributes: _Attributes | None = None,
        context: Context | None = None,
    ) -> None:
        # Defensive copy of the attributes mapping so later mutation
        # of the caller's dict cannot rewrite recorded history.
        recorded_attrs = dict(attributes) if attributes is not None else None
        self.calls.append((amount, recorded_attrs, context))


def observed_value(counter: _Counter) -> int | float:
    """Return the cumulative ``add`` total for a fake counter.

    Test-only helper. Raises ``TypeError`` if ``counter`` is not a
    ``_FakeCounter`` — production OTel counters do not expose
    observation, so a misuse (test running against a
    ``build_sessions_telemetry`` that was wired with a real meter)
    fails loudly rather than producing a confusing attribute error.
    """
    if not isinstance(counter, _FakeCounter):
        raise TypeError(
            f"observed_value: expected _FakeCounter, got "
            f"{type(counter).__name__}. Tests must call "
            f"build_sessions_telemetry() without a meter argument so "
            f"the container is populated with fake counters."
        )
    return sum(amount for amount, _attrs, _context in counter.calls)


@dataclass(frozen=True, slots=True)
class _SessionsTelemetry:
    """Container for the named counters introduced by composer progress
    persistence. All counters default to fakes so tests can assert without
    wiring the real OTel SDK; production wiring replaces them at startup.
    """

    # Phase 1 audit-primacy counters only. Phase 2 (redaction) adds
    # ``summarizer_errors_total`` and ``unknown_response_key_total``;
    # Phase 3 (compose loop + audit-grade view) adds
    # ``tool_call_cap_exceeded_total`` and ``audit_grade_view_total``.
    # Each phase extends this dataclass when its emitter ships,
    # which keeps "registered" and "exercised" in lock-step
    # operationally.
    tool_row_tier1_violation_total: _Counter
    state_rolled_back_during_persist_total: _Counter
    tool_row_persist_failed_during_unwind_total: _Counter
    tool_row_integrity_violation_total: _Counter


def build_sessions_telemetry(
    *, meter: _Meter | None = None
) -> _SessionsTelemetry:
    """Build a telemetry container.

    With ``meter=None`` (the default) returns ``_FakeCounter``
    instances; tests use this path. Production callers pass an OTel
    ``Meter`` (typed structurally as ``_Meter`` so we don't import
    ``opentelemetry.metrics`` here unnecessarily — the structural
    Protocol is satisfied by the real meter at runtime).
    """

    if meter is None:
        return _SessionsTelemetry(
            tool_row_tier1_violation_total=_FakeCounter(),
            state_rolled_back_during_persist_total=_FakeCounter(),
            tool_row_persist_failed_during_unwind_total=_FakeCounter(),
            tool_row_integrity_violation_total=_FakeCounter(),
        )

    # Production wiring against the real OTel meter. The ``_Meter``
    # Protocol satisfies mypy without ``# type: ignore`` decorations
    # — the real OTel ``Meter.create_counter`` matches the structural
    # contract, and the returned counter satisfies ``_Counter``.
    return _SessionsTelemetry(
        tool_row_tier1_violation_total=meter.create_counter(
            "composer.audit.tool_row_tier1_violation_total"
        ),
        state_rolled_back_during_persist_total=meter.create_counter(
            "composer.audit.state_rolled_back_during_persist_total"
        ),
        tool_row_persist_failed_during_unwind_total=meter.create_counter(
            "composer.audit.tool_row_persist_failed_during_unwind_total"
        ),
        tool_row_integrity_violation_total=meter.create_counter(
            "composer.audit.tool_row_integrity_violation_total"
        ),
    )
```

- [ ] **Step 4: Run tests to verify pass**

```bash
.venv/bin/python -m pytest tests/unit/web/sessions/test_telemetry.py -v
```
Expected: PASS.

- [ ] **Step 5: Write the failing constructor-signature test**

Create `tests/unit/web/sessions/test_service_construction.py`:

```python
"""Tests pinning the SessionServiceImpl constructor signature.

Phase 1 extends the constructor with required ``telemetry`` and ``log``
arguments so that ``persist_compose_turn`` and the audit-failure
disposition path can emit OTel counters and (only when the audit
system itself fails) log diagnostics. The signature is part of the
service's public contract; this test prevents accidental drift.
"""
from __future__ import annotations

from sqlalchemy.pool import StaticPool

import pytest
import structlog

from elspeth.web.sessions.telemetry import build_sessions_telemetry
from elspeth.web.sessions.engine import create_session_engine
from elspeth.web.sessions.schema import initialize_session_schema
from elspeth.web.sessions.service import SessionServiceImpl


@pytest.fixture
def engine():
    eng = create_session_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    initialize_session_schema(eng)
    return eng


def test_constructor_accepts_telemetry_and_log(engine, tmp_path):
    telem = build_sessions_telemetry()
    log = structlog.get_logger("test")
    service = SessionServiceImpl(
        engine,
        data_dir=tmp_path,
        telemetry=telem,
        log=log,
    )
    assert service._telemetry is telem
    assert service._log is log


def test_constructor_rejects_missing_telemetry(engine, tmp_path):
    with pytest.raises(TypeError, match="telemetry"):
        SessionServiceImpl(engine, data_dir=tmp_path)  # type: ignore[call-arg]


def test_constructor_rejects_missing_log(engine, tmp_path):
    telem = build_sessions_telemetry()
    with pytest.raises(TypeError, match="log"):
        SessionServiceImpl(engine, data_dir=tmp_path, telemetry=telem)  # type: ignore[call-arg]
```

- [ ] **Step 6: Run the constructor test to verify it fails**

```bash
.venv/bin/python -m pytest tests/unit/web/sessions/test_service_construction.py -v
```
Expected: FAIL — constructor does not yet accept `telemetry` or `log`.

- [ ] **Step 7: Extend `SessionServiceImpl.__init__`**

In `src/elspeth/web/sessions/service.py`, replace the existing
constructor (around line 86) with:

```python
from elspeth.web.sessions.telemetry import _SessionsTelemetry  # add to imports

class SessionServiceImpl:
    """Concrete session service backed by SQLAlchemy Core.

    All public methods are async. Database I/O runs through _run_sync() in a
    bounded worker thread so the async event loop is never blocked.
    """

    def __init__(
        self,
        engine: Engine,
        data_dir: Path | None = None,
        *,
        telemetry: _SessionsTelemetry,
        log: structlog.stdlib.BoundLogger,
    ) -> None:
        self._engine = engine
        self._data_dir = data_dir
        self._telemetry = telemetry
        self._log = log
```

Notes:
- `telemetry` and `log` are keyword-only (after the `*,` separator) so
  positional callers cannot accidentally swap them.
- `data_dir` keeps its default for backwards compatibility within this
  PR's call-site sweep — production passes it positionally (per
  app.py:391 today). After the sweep, every call site supplies
  `telemetry` and `log` explicitly.
- The `_SessionsTelemetry` import uses the underscore-prefixed type:
  the type itself is module-private but importing it here is
  intentional because `service.py` needs to type-check against the
  concrete container shape, not a generic `object`.
- **Test-only access convention.** The Phase 1 unit and integration
  tests access ``service._engine`` directly to set up fixture state
  (insert sessions, run schema-introspection queries) that the
  public ``SessionServiceImpl`` API does not expose. The underscore
  is conventional Python for "internal," not "do not access from
  tests" — pytest-style fixtures and tests have legitimate reasons
  to reach past the public surface. If a future Phase introduces a
  third consumer that ALSO needs raw engine access (a maintenance
  CLI, an admin tool), promote the field to a typed
  ``engine_for_test_or_admin: Engine`` property. Until then,
  ``service._engine`` is the documented test pattern and need not
  be re-exposed. Closes synthesised review finding SA-12 / M12.

You will also need to add `import structlog` near the top of the
module if it is not already imported. **As of the Phase 1 plan-review
verification, `structlog` is NOT imported by
`src/elspeth/web/sessions/service.py` — this Step MUST add the import**
(alongside the standard-library imports at the top of the file). Confirm
via `grep -n '^import structlog' src/elspeth/web/sessions/service.py`
before editing.

- [ ] **Step 8: Update production wiring at `app.py:391`**

Replace the existing line at `src/elspeth/web/app.py:391`:

```python
# BEFORE
session_service = SessionServiceImpl(session_engine, data_dir=settings.data_dir)
```

with:

```python
# AFTER
from opentelemetry import metrics  # add to top-of-module imports — NOT currently imported by app.py per plan-review verification; required for the metrics.get_meter(...) call below
from elspeth.web.sessions.telemetry import build_sessions_telemetry  # add to imports — also new

# ...

session_service = SessionServiceImpl(
    session_engine,
    data_dir=settings.data_dir,
    telemetry=build_sessions_telemetry(meter=metrics.get_meter("elspeth.web.composer")),
    log=structlog.get_logger("sessions"),
)
```

Notes:
- `metrics.get_meter(...)` returns a real OTel meter when the runtime
  has a `MeterProvider` configured, and a no-op meter otherwise.
  Production code never uses fake counters — fakes are a test affordance
  only (per `_FakeCounter` docstring in telemetry.py).
- `structlog.get_logger("sessions")` matches the established
  pattern in `app.py` for naming subsystem loggers.

- [ ] **Step 9: Sweep every existing constructor call site**

Run the inventory command before editing:

```bash
rg -n "SessionServiceImpl\\(" src tests -g '*.py'
```

Expected before this task on the reviewed snapshot: 14 call sites,
including:

- `src/elspeth/web/app.py`
- `tests/unit/web/sessions/test_fork.py`
- `tests/unit/web/sessions/test_datetime_timezone.py`
- `tests/unit/web/sessions/test_routes.py`
- `tests/unit/web/sessions/test_service.py`
- `tests/unit/web/blobs/test_routes.py`

Update every construction to pass `telemetry=build_sessions_telemetry()`
and `log=structlog.get_logger(...)` in tests, or the production OTel
meter/logger in `app.py`. Re-run the inventory and paste both counts
into the PR description. Any remaining `SessionServiceImpl(engine)`
shape is a blocker.

- [ ] **Step 10: Run all affected tests + mypy**

```bash
.venv/bin/python -m pytest tests/unit/web/sessions/test_telemetry.py tests/unit/web/sessions/test_service_construction.py tests/unit/web/sessions/test_service.py tests/unit/web/sessions/test_routes.py tests/unit/web/sessions/test_fork.py tests/unit/web/sessions/test_datetime_timezone.py tests/unit/web/blobs/test_routes.py -v
.venv/bin/python -m mypy src/elspeth/web/sessions/service.py src/elspeth/web/app.py src/elspeth/web/sessions/telemetry.py
```
Expected: all tests PASS; mypy clean.

- [ ] **Step 11: Commit**

```bash
git add src/elspeth/web/sessions/telemetry.py \
        tests/unit/web/sessions/test_telemetry.py \
        tests/unit/web/sessions/test_service_construction.py \
        tests/unit/web/sessions/test_service.py \
        tests/unit/web/sessions/test_routes.py \
        tests/unit/web/sessions/test_fork.py \
        tests/unit/web/sessions/test_datetime_timezone.py \
        tests/unit/web/blobs/test_routes.py \
        src/elspeth/web/sessions/service.py \
        src/elspeth/web/app.py
git commit -m "feat(sessions): add SessionsTelemetry counter container and wire into SessionServiceImpl (composer-progress-persistence phase 1)

- Adds web/sessions/telemetry.py with the _SessionsTelemetry container
  and build_sessions_telemetry() factory.
- Extends SessionServiceImpl.__init__ to accept required telemetry and
  log keyword arguments; production wiring at app.py:391 updated in the
  same commit (no-legacy single-cut policy).
- Production wiring uses opentelemetry.metrics.get_meter(...) for the
  real OTel meter and structlog.get_logger('sessions') for the log.
"
```

---

## Task 6: Persist-payload dataclasses

**Files:**
- Create: `src/elspeth/web/sessions/_persist_payload.py`
- Create: `tests/unit/web/sessions/test_persist_payload.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/web/sessions/test_persist_payload.py`:

```python
"""Tests for the persist-payload dataclasses (spec §5.2.1)."""
from __future__ import annotations

import pytest

from elspeth.web.sessions._persist_payload import (
    _AuditOutcome,
    _RedactedToolRow,
    _StatePayload,
    _ToolOutcome,
)


def test_audit_outcome_success_shape():
    """Success: assistant_id set, no unwind failure."""
    outcome = _AuditOutcome(
        assistant_id="abc",
        unwind_audit_failed=False,
    )
    assert outcome.assistant_id == "abc"
    assert outcome.unwind_audit_failed is False


def test_audit_outcome_unwind_failure_shape():
    """Tool failed AND audit unwind failed: assistant_id=None,
    flag set. Caller will raise the captured plugin crash."""
    outcome = _AuditOutcome(
        assistant_id=None,
        unwind_audit_failed=True,
    )
    assert outcome.assistant_id is None
    assert outcome.unwind_audit_failed is True


def test_audit_outcome_rejects_ambiguous_shape():
    """assistant_id=set + unwind_audit_failed=True is contradictory:
    the unwind path runs only when the tool already failed, so no
    assistant message could have been produced. The dataclass rejects
    the combination at construction time."""
    with pytest.raises(ValueError, match="incompatible"):
        _AuditOutcome(
            assistant_id="abc",          # produced by a successful path
            unwind_audit_failed=True,    # claimed by an unwind path
        )


def test_audit_outcome_no_tier1_violation_field():
    """Sanity: the tier1_violation flag-return path was deleted in
    Stage 4 of the plan revision; persist_compose_turn now raises
    AuditIntegrityError directly. Closes finding H1."""
    import dataclasses
    fields = {f.name for f in dataclasses.fields(_AuditOutcome)}
    assert fields == {"assistant_id", "unwind_audit_failed"}


def test_redacted_tool_row_with_state_advance():
    from elspeth.web.sessions.protocol import CompositionStateData

    row = _RedactedToolRow(
        tool_call_id="tc_1",
        content='{"ok": true}',
        composition_state_payload=_StatePayload(
            # B1 (Phase 1 plan-review synthesis): no ``version=``.
            # Version is allocated inside _session_write_lock by
            # ``_insert_composition_state`` (Task 10), not supplied by
            # the caller. Removing the field at the dataclass level
            # forecloses the dual-allocator race that fabricated
            # Tier-1 violations on contention loss.
            data=CompositionStateData(
                source={"kind": "test"},
                nodes=[],
                edges=[],
                outputs=[],
                metadata_={},
                is_valid=True,
                validation_errors=None,
            ),
            derived_from_state_id="prev_state_id",
        ),
    )
    assert row.composition_state_payload is not None
    assert row.composition_state_payload.data.is_valid is True
    assert row.composition_state_payload.derived_from_state_id == "prev_state_id"


def test_state_payload_has_no_version_field():
    """B1 (Phase 1 plan-review synthesis): ``_StatePayload`` MUST NOT
    carry a caller-supplied ``version`` field. Version is allocated
    inside _session_write_lock by
    ``_insert_composition_state`` via
    ``SELECT COALESCE(MAX(version), 0) + 1 FROM composition_states
    WHERE session_id = :sid`` (Task 10).

    Pre-B1 the field existed; the compose loop in Phase 3 read
    ``MAX(version)`` outside the lock and dispatched a precomputed
    version into the locked helper. Two concurrent allocators could
    both compute ``MAX+1``; the loser's INSERT triggered
    ``uq_composition_state_version`` → ``IntegrityError`` → the
    locked-path handler incremented ``tool_row_integrity_violation_total``
    on what was structurally a contention loss, fabricating a Tier-1
    audit-integrity violation. SLO threshold is 0; the alert fires on a
    non-event.

    The fix is structural — version simply isn't a payload field — so
    no new caller can reintroduce the race by accident. This test pins
    the contract so a refactor that re-adds the field fails fast."""
    import dataclasses
    fields = {f.name for f in dataclasses.fields(_StatePayload)}
    assert "version" not in fields, (
        "B1 regression: _StatePayload must not carry a caller-supplied "
        "version field — version is allocated by "
        "_insert_composition_state under _session_write_lock"
    )
    assert fields == {"data", "derived_from_state_id"}, (
        f"unexpected _StatePayload fields: {fields}"
    )


def test_tool_outcome_state_unchanged_when_pre_eq_post():
    outcome = _ToolOutcome(
        call=type("FakeCall", (), {"id": "x", "function": type("F", (), {"name": "n"})()})(),
        response={"ok": True},
        error_class=None,
        error_message=None,
        pre_version=3,
        post_version=3,
    )
    assert outcome.post_version == outcome.pre_version


def test_tool_outcome_freezes_dict_response():
    """``_ToolOutcome.__post_init__`` must call ``freeze_fields`` on
    ``call`` and ``response`` so that a dict response cannot be mutated
    through the dataclass reference. Without the guard, ``frozen=True``
    is a lie about deep immutability — the attribute cannot be
    reassigned but the dict it points to remains fully mutable.
    Closes synthesised review finding H5."""
    from types import MappingProxyType

    response_dict = {"ok": True, "nested": {"k": "v"}}
    outcome = _ToolOutcome(
        call={"id": "tc_x", "function": {"name": "set_source"}},
        response=response_dict,
        error_class=None,
        error_message=None,
        pre_version=1,
        post_version=2,
    )

    # Both call and response must be deeply frozen — the outer mapping
    # is a MappingProxyType (rejects __setitem__) and nested dicts are
    # also frozen.
    assert isinstance(outcome.response, MappingProxyType)
    assert isinstance(outcome.call, MappingProxyType)
    with pytest.raises(TypeError):
        outcome.response["ok"] = False  # type: ignore[index]
    # Mutation via the original reference must not be visible through
    # the dataclass reference (deep_freeze copies inputs to immutable
    # equivalents; the original dict and the proxy are distinct).
    response_dict["ok"] = False
    assert outcome.response["ok"] is True
```

- [ ] **Step 2: Run test to verify it fails**

```bash
.venv/bin/python -m pytest tests/unit/web/sessions/test_persist_payload.py -v
```
Expected: FAIL — module does not exist.

- [ ] **Step 3: Implement the dataclasses**

Create `src/elspeth/web/sessions/_persist_payload.py`:

```python
"""Dataclasses passed across the async/sync boundary in
``SessionServiceImpl.persist_compose_turn`` (spec §5.2.1).

These types have no async behaviour; they are pure data containers that
the compose loop populates in async land and then hands to the sync
worker via ``_run_sync``.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from elspeth.contracts.freeze import freeze_fields
from elspeth.web.sessions.protocol import CompositionStateData


@dataclass(frozen=True, slots=True)
class _StatePayload:
    """Snapshot of a CompositionState ready for insertion.

    Composes the existing :class:`CompositionStateData` input DTO (which
    carries the per-column state contents and is already
    ``freeze_fields``-protected by its own ``__post_init__``) with a
    ``derived_from_state_id`` that records lineage from the pre-call
    state.

    **B1 (Phase 1 plan-review synthesis): no ``version`` field.** Earlier
    drafts of this dataclass carried a caller-supplied ``version: int``.
    That contract was unsafe: in Phase 3 the compose loop reads
    ``MAX(version)`` outside the session write lock and then
    dispatches into the ``_session_write_lock``-protected ``_insert_composition_state``
    helper. Two concurrent allocators can both compute ``MAX+1`` before
    either acquires the lock; the loser's INSERT then hits
    ``uq_composition_state_version`` and the locked path's
    ``IntegrityError`` handler classifies it as a Tier-1 audit-integrity
    violation — fabricating a Tier-1 violation from a benign contention
    loss. SLO threshold for ``tool_row_integrity_violation_total`` is 0,
    so the alert fires on a non-event. Under ELSPETH's auditability
    standard this is evidence-tampering-class harm: the audit trail
    asserts a violation that did not occur.

    The fix is structural: the version is no longer a payload field at
    all. ``_insert_composition_state`` allocates it under the held
    session write lock via ``SELECT COALESCE(MAX(version), 0) + 1 FROM
    composition_states WHERE session_id = :sid`` (see Task 10). With
    version off the payload, the dual-allocator race becomes
    structurally impossible: every caller must be inside the lock context to invoke
    the helper, and the SELECT-MAX-then-INSERT sequence is atomic
    against every other writer for that session.

    The contract is fixed in Phase 1 even though the call shape only
    manifests in Phase 3 — making the helper impossible to misuse from
    Phase 3 onward is cheaper than catching the misuse later.

    Why this shape rather than a single JSON blob:

    The ``composition_states`` table has eight content columns
    (``source/nodes/edges/outputs/metadata_/is_valid/validation_errors/derived_from_state_id``).
    The plan's earlier ``payload_json: str`` design was a hallucination —
    no ``payload`` column exists, and the existing
    ``save_composition_state`` insert at ``service.py:395-418`` writes
    each column individually via a method-local ``_enveloped(...)`` helper
    and ``deep_thaw(...)`` patterns. Task 10 extracts that rule to the
    shared ``_enveloped_state_column(...)`` helper. ``_StatePayload``
    mirrors that real schema by reusing :class:`CompositionStateData` rather than
    duplicating its fields and freeze-guard machinery.

    ``derived_from_state_id`` is ``str | None`` rather than ``str``
    because the existing inline inserts at ``service.py:395-418``
    (initial state) currently set it to ``None``. The compose-loop
    caller in Phase 3 will always supply a non-None value (every
    tool-call-driven state advance has a predecessor).

    Note on freeze-fields. ``derived_from_state_id`` is a scalar (or
    ``None``); ``data`` is a frozen ``CompositionStateData`` with its own
    ``freeze_fields`` discipline. No ``__post_init__`` is required on
    ``_StatePayload`` itself — ``frozen=True`` is sufficient because
    every remaining field is either scalar or an already-frozen
    dataclass. (Removing ``version`` did not change this analysis;
    ``version: int`` was scalar too.)
    """

    data: CompositionStateData
    derived_from_state_id: str | None = None


@dataclass(frozen=True, slots=True)
class _ToolOutcome:
    """Result of one tool call within a compose turn.

    The ``call`` and ``response`` fields are typed ``Any`` because the
    compose loop populates them with framework-specific objects (LiteLLM
    ToolCall, Pydantic response models, plain dicts, etc.) that this
    module deliberately does not couple to. At runtime these values are
    typically dicts or ``Mapping`` types, so the ``frozen=True``
    declaration alone is a lie about immutability — the dataclass
    attribute cannot be reassigned, but the dict it points to remains
    fully mutable through the reference.

    CLAUDE.md's ``freeze_fields`` contract is unconditional for frozen
    dataclasses with container/Any fields: ``__post_init__`` must call
    ``freeze_fields`` on every such field. ``deep_freeze`` (which
    ``freeze_fields`` invokes per field) is identity-preserving for
    values that are already frozen, so the cost of running it on
    already-immutable inputs (e.g. an integer-only ``call``, which
    won't happen in practice but is contractually possible) is zero.
    """

    call: Any                       # ToolCall — typed in protocol module
    response: Any                   # tool response object or None on error
    error_class: str | None
    error_message: str | None
    pre_version: int
    post_version: int

    def __post_init__(self) -> None:
        freeze_fields(self, "call", "response")


@dataclass(frozen=True, slots=True)
class _RedactedToolRow:
    """One persisted tool row, with redactions already applied."""

    tool_call_id: str
    content: str                                       # JSON-serialised redacted response
    composition_state_payload: _StatePayload | None    # set iff state advanced


@dataclass(frozen=True, slots=True)
class _AuditOutcome:
    """Disposition returned by SessionServiceImpl.persist_compose_turn (§5.2.2).

    Two outcome shapes:

    - **Success.** ``assistant_id`` is set, ``unwind_audit_failed=False``.
      Caller continues with the new assistant message id.
    - **Tool failed AND audit unwind failed.** ``assistant_id=None``,
      ``unwind_audit_failed=True``. Caller raises the captured plugin
      crash; the audit failure is recorded by ``persist_compose_turn``
      via counter increment + ``slog.warning`` (permitted under
      CLAUDE.md primacy because the audit system itself failed).

    There is NO tier-1-violation outcome shape. When the audit
    database fails AND no plugin crash is in flight,
    ``persist_compose_turn`` raises
    :class:`elspeth.contracts.errors.AuditIntegrityError` directly,
    chained from the underlying ``OperationalError`` via ``raise ...
    from audit_exc``. The exception is registered in
    ``TIER_1_ERRORS`` (via the ``@tier_1_error`` decoration on
    ``AuditIntegrityError``) so ``except Exception:`` blocks cannot
    silently swallow it. The caller has no opportunity to ignore the
    failure — this is the Tier-1 crash doctrine ("Bad data in the
    audit trail = crash immediately") encoded structurally rather
    than asked nicely.

    Why ``unwind_audit_failed`` stays a flag-return (not a raise):
    when a tool plugin has crashed in flight, the caller already has
    a captured plugin-crash exception to raise. Surfacing a separate
    audit exception would mask the original tool failure. The flag
    tells the caller "your raise should ALSO record this audit
    failure," and the counter + slog inside ``persist_compose_turn``
    have already done so.

    Closes synthesised review finding H1 (audit primacy via
    return-flag instead of raised exception violates Tier-1 doctrine).
    """

    assistant_id: str | None
    unwind_audit_failed: bool

    def __post_init__(self) -> None:
        # Success and unwind-failure are the only two valid shapes.
        # Reject any combination that would make the outcome ambiguous.
        if self.assistant_id is not None and self.unwind_audit_failed:
            raise ValueError(
                "_AuditOutcome: unwind_audit_failed=True is incompatible with "
                "assistant_id being set; the unwind path cannot have produced "
                "an assistant id"
            )
```

- [ ] **Step 4: Run tests to verify pass**

```bash
.venv/bin/python -m pytest tests/unit/web/sessions/test_persist_payload.py -v
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/elspeth/web/sessions/_persist_payload.py tests/unit/web/sessions/test_persist_payload.py
git commit -m "feat(sessions): add persist-payload dataclasses (composer-progress-persistence phase 1)"
```

---

## Task 7: session write-lock helpers (`_session_write_lock` + PostgreSQL advisory lock)

**B3 (Phase 1 plan-review synthesis) — two-argument advisory-lock form.**
The synthesised plan review flagged the single-argument form
`pg_advisory_xact_lock(hashtextextended(session_id, 0))` as
cross-application-unsafe: PostgreSQL's single-argument advisory locks
all live in one cluster-wide namespace, so any other application
on the same Postgres cluster — or any other ELSPETH subsystem
adding its own single-argument advisory lock in the future — could
collide on the same hash slot, producing spurious contention or, in
the pathological case, a circular wait between unrelated subsystems
holding each other's locks. Phase 1 closes the gap by switching to
the two-argument form
`pg_advisory_xact_lock(<classid>, hashtext(session_id))`,
where `<classid>` is a stable 32-bit constant defined in
`src/elspeth/contracts/advisory_locks.py` and unique to the sessions
DB lock. The constant is **on-the-wire ABI** under change control:
two ELSPETH instances on the same Postgres cluster MUST use the same
classid or they will not serialise against each other.

**Spec-vs-plan note.** Spec §5.7.1 currently shows the
single-argument form (the spec was written before the B3 finding).
The two-argument form here supersedes the spec for Phase 1; the
spec-amendment follow-up is tracked as a Filigree observation
(see Task 19's spec-drift entry). The implementer should not
"reconcile" this divergence by reverting to the spec form — the
spec amendment is what is pending, not the plan.

**Postgres signature note.** `pg_advisory_xact_lock(int, int)` takes
two signed `int4` arguments. Use `hashtext(text)` for the session key
because it returns the required `int4` directly. Do **not** use
`hashtextextended(... )::int`: PostgreSQL casts `bigint` to `integer`
with range checking, not silent truncation, so that expression can
raise on ordinary hash values before the lock is acquired. `hashtext`
still produces a 32-bit hash slot, so birthday collisions become
probable around ~65k *concurrent* sessions hashing to the same classid.
This is **operationally acceptable** because:

1. Collisions cause two unrelated `session_id` values to share the
   same advisory-lock slot, producing benign extra serialisation —
   never duplicate rows, never lost writes. The unique index
   `ix_chat_messages_session_sequence` is the correctness guarantee;
   the advisory lock is a contention-reducer ahead of it.
2. The 65k-concurrent-session ceiling refers to *active concurrent
   writers* per Postgres cluster, not the lifetime session count.
   A deployment with 65k simultaneously-writing sessions is a
   capacity-planning concern that dwarfs the lock-collision concern.
3. The spec amendment task in this phase must update every stale
   `hashtextextended(... )::int` snippet to `hashtext(...)`. Until that
   amendment lands, this plan is authoritative for Phase 1 lock SQL.

The collision-space prose elsewhere in this plan (Task 16 module
docstring, Task 16 fixture comments) has been updated to reflect
this truncation honestly.

**SQLite serialization amendment.** The reviewed plan's older SQLite
path treated `_acquire_session_advisory_lock` as a no-op and relied on
"SQLite global write lock" prose. That is not mechanical enough for
the current service: `_run_sync` dispatches writes through separate
worker executions, and sequence/version allocation uses `SELECT MAX`
before insert. Phase 1 must serialize same-session SQLite writers in
process before the allocator reads. The plan therefore introduces a
single context manager, `_session_write_lock(conn, session_id)`, used
by every session-scoped writer:

- PostgreSQL branch: call `_acquire_session_advisory_lock(conn, session_id)` and yield.
- SQLite branch: acquire a process-wide reentrant lock keyed by `(database_url, session_id)`, then yield for the whole allocator + insert block.

This is sufficient for the current staging shape (one uvicorn process
against one SQLite file) and is backed by explicit same-session
concurrency tests. If staging becomes multi-process SQLite, Phase 1
must switch SQLite to a database-level `BEGIN IMMEDIATE`/busy-timeout
strategy before deploy; do not rely on the process lock across
processes. The key includes the engine URL so unrelated SQLite databases
inside the same process do not serialize each other's sessions.

**Files:**
- Create: `src/elspeth/contracts/advisory_locks.py`
- Modify: `src/elspeth/web/sessions/service.py`
- Test: `tests/unit/web/sessions/test_persist_compose_turn.py` (create)

- [ ] **Step 1: Create the advisory-lock classid registry**

Create `src/elspeth/contracts/advisory_locks.py`:

```python
"""PostgreSQL advisory-lock classid registry.

PostgreSQL exposes two flavours of advisory locks: the
single-argument form (one int8 namespace, cluster-wide) and the
two-argument form (two int4 namespaces, also cluster-wide but
partitioned by the first argument — the *classid*). ELSPETH uses
the two-argument form exclusively so that each subsystem holding
advisory locks gets its own classid namespace, avoiding cross-
subsystem collision (and cross-application collision with any
other software using single-argument advisory locks on the same
cluster).

This module is the SINGLE registry of classid values. Adding a
new advisory lock anywhere in ELSPETH MUST add a new constant
here with a distinct value. Reusing a classid across subsystems
re-introduces the collision risk that splitting the namespace
was meant to eliminate.

ABI commitment
--------------
Every constant defined here is **on-the-wire ABI**. Two ELSPETH
instances on the same Postgres cluster — including instances
running different ELSPETH versions during a rolling deploy —
MUST agree on the value of every classid in this module, or
they will not serialise against each other on the same logical
resource. A version mismatch on a classid value produces a silent
correctness violation: both instances think they hold the lock,
both execute the protected code path concurrently, and the
correctness guarantee the lock was protecting is lost.

Changing any constant here therefore requires:

1. An ADR documenting the rationale and the migration plan.
2. A coordinated deploy that drains all writers using the old
   value before any writer using the new value comes online.
3. A schema/runbook update so operators understand the
   constraint.

The 32-bit signed integer space is enormous (~4.3 billion
values); pick distinct values, never reuse a retired value
within the same major release.
"""
from __future__ import annotations

# 0x454C5350 = 1,162,629,968 — ASCII "ELSP" big-endian. First
# classid assigned in this registry; chosen so a Postgres operator
# inspecting pg_locks sees a recognisable value rather than a random
# magic number. Used by SessionServiceImpl._acquire_session_advisory_lock
# (src/elspeth/web/sessions/service.py) for the session-scoped write
# lock that serialises persist_compose_turn / save_composition_state /
# set_active_state writers within a single Postgres cluster.
ELSPETH_SESSIONS_LOCK_CLASSID: int = 0x454C5350
```

- [ ] **Step 2: Write the failing test**

Create `tests/unit/web/sessions/test_persist_compose_turn.py`:

```python
"""Unit tests for SessionServiceImpl persistence helpers (spec §5.7.1).

Uses the shared ``engine`` fixture and ``_make_session`` helper from
``tests/unit/web/conftest.py``.
"""
from __future__ import annotations

from datetime import UTC, datetime

import pytest
from sqlalchemy import text

from elspeth.web.sessions.service import SessionServiceImpl

from tests.unit.web.conftest import _make_session


@pytest.fixture
def service(engine, tmp_path):
    """Use the shared in-memory SQLite ``engine`` fixture from
    ``tests/unit/web/conftest.py`` (which already wires
    ``create_session_engine`` + ``StaticPool`` + schema bootstrap).
    """
    from elspeth.web.sessions.telemetry import build_sessions_telemetry
    import structlog

    return SessionServiceImpl(
        engine,
        data_dir=tmp_path,
        telemetry=build_sessions_telemetry(),
        log=structlog.get_logger(),
    )


def test_advisory_lock_sqlite_is_noop(service):
    """SQLite does not support pg_advisory_xact_lock; the Postgres-only
    helper itself remains a no-op. Same-session SQLite serialization is
    verified through _session_write_lock tests below."""
    with service._engine.begin() as conn:
        # No raise expected.
        service._acquire_session_advisory_lock(conn, "session_1")


def test_session_write_lock_sqlite_is_reentrant(service):
    """SQLite branch uses a process-wide per-session RLock so nested
    helper calls inside one transaction cannot deadlock."""
    with service._engine.begin() as conn:
        with service._session_write_lock(conn, "session_1"):
            with service._session_write_lock(conn, "session_1"):
                pass
```

- [ ] **Step 3: Run test to verify it fails**

```bash
.venv/bin/python -m pytest tests/unit/web/sessions/test_persist_compose_turn.py::test_advisory_lock_sqlite_is_noop -v
```
Expected: FAIL — `_acquire_session_advisory_lock` does not exist.

- [ ] **Step 4: Implement the helper**

In `src/elspeth/web/sessions/service.py`, add the helpers inside the `SessionServiceImpl` class. Note the import of `ELSPETH_SESSIONS_LOCK_CLASSID` at module top — cite the constant by name at every call site so future grep finds them all (open-coded literals defeat the registry pattern). Add `import contextlib` and `import threading`, and define the SQLite lock registry at module top so every service instance in the process serializes on the same locks:

```python
_SQLITE_SESSION_LOCKS_GUARD = threading.RLock()
_SQLITE_SESSION_LOCKS: dict[tuple[str, str], threading.RLock] = {}
```

```python
# At module top of src/elspeth/web/sessions/service.py:
from elspeth.contracts.advisory_locks import ELSPETH_SESSIONS_LOCK_CLASSID

# Inside SessionServiceImpl:
def _acquire_session_advisory_lock(self, conn: Connection, session_id: str) -> None:
    """Acquire a session write lock for the duration of the
    current transaction. Released automatically on COMMIT or ROLLBACK.

    SQLite: no-op. SQLite serialization is owned by
    ``_session_write_lock`` below; this helper exists only for the
    PostgreSQL advisory-lock SQL and remains no-op on SQLite so callers
    can test the dialect-specific SQL separately.

    PostgreSQL: pg_advisory_xact_lock(ELSPETH_SESSIONS_LOCK_CLASSID,
    hashtext(session_id)) — the **two-argument** form
    (B3 from the Phase 1 plan-review synthesis). The classid namespace
    is reserved in src/elspeth/contracts/advisory_locks.py and is
    on-the-wire ABI under change control; do not open-code the literal
    here, always import the constant.

    Hash-function notes:

    * `pg_advisory_xact_lock(int, int)` requires two signed int4
      arguments. `hashtext(text)` returns int4 directly. Do not use
      `hashtextextended(... )::int`: PostgreSQL integer casts are
      range-checked and may fail before the lock is acquired.
    * Birthday collisions become probable around ~65k *concurrent*
      sessions hashing to the same classid slot. This
      is benign — the unique index ix_chat_messages_session_sequence
      is the correctness guarantee; the advisory lock is a
      contention-reducer ahead of it. Collisions cause spurious
      serialisation between two unrelated sessions, never duplicate
      rows or lost writes.
    * The classid value is **NOT** a deployment knob. Two ELSPETH
      instances on the same Postgres cluster (including different
      versions during a rolling deploy) MUST share the same value
      or they will not mutually exclude each other. See
      src/elspeth/contracts/advisory_locks.py for the ABI commitment.
    """
    dialect = self._engine.dialect.name
    if dialect == "sqlite":
        return  # SQLite serialization is owned by _session_write_lock
    if dialect == "postgresql":
        conn.exec_driver_sql(
            "SELECT pg_advisory_xact_lock(%s, hashtext(%s))",
            (ELSPETH_SESSIONS_LOCK_CLASSID, session_id),
        )
        return
    raise NotImplementedError(
        f"_acquire_session_advisory_lock not implemented for dialect {dialect}"
    )


def _sqlite_lock_for_session(self, session_id: str) -> threading.RLock:
    """Return the process-wide SQLite write lock for one DB/session pair."""
    key = (str(self._engine.url), session_id)
    with _SQLITE_SESSION_LOCKS_GUARD:
        lock = _SQLITE_SESSION_LOCKS.get(key)
        if lock is None:
            lock = threading.RLock()
            _SQLITE_SESSION_LOCKS[key] = lock
        return lock


@contextlib.contextmanager
def _session_write_lock(self, conn: Connection, session_id: str):
    """Serialize same-session sequence/version allocators.

    PostgreSQL uses the transaction-scoped advisory lock. SQLite uses a
    process-wide per-session RLock around the whole allocator + insert
    sequence. Every caller that performs ``SELECT MAX(...) + 1`` for
    ``chat_messages.sequence_no`` or ``composition_states.version`` MUST
    wrap that read and every dependent INSERT in this context.
    """
    dialect = self._engine.dialect.name
    if dialect == "sqlite":
        with self._sqlite_lock_for_session(session_id):
            yield
        return
    if dialect == "postgresql":
        self._acquire_session_advisory_lock(conn, session_id)
        yield
        return
    raise NotImplementedError(
        f"_session_write_lock not implemented for dialect {dialect}"
    )
```

- [ ] **Step 5: Run tests to verify pass**

```bash
.venv/bin/python -m pytest tests/unit/web/sessions/test_persist_compose_turn.py -v -k "advisory_lock_sqlite_is_noop or session_write_lock_sqlite_is_reentrant"
```
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/elspeth/contracts/advisory_locks.py src/elspeth/web/sessions/service.py tests/unit/web/sessions/test_persist_compose_turn.py
git commit -m "feat(sessions): add session write-lock helpers with Postgres advisory namespace and SQLite per-session lock (composer-progress-persistence phase 1)"
```

---

## Task 8: `_reserve_sequence_range` helper

**Files:**
- Modify: `src/elspeth/web/sessions/service.py`
- Test: `tests/unit/web/sessions/test_persist_compose_turn.py` (extend)

- [ ] **Step 1: Write the failing test**

Add to `tests/unit/web/sessions/test_persist_compose_turn.py`:

```python
def test_reserve_sequence_range_starts_at_one_for_empty_session(service):
    with service._engine.begin() as conn:
        _make_session(conn, session_id="s1")
        base = service._reserve_sequence_range(conn, "s1", count=3)
        assert base == 1


def test_reserve_sequence_range_continues_after_existing(service):
    from elspeth.web.sessions import models
    from sqlalchemy import insert
    with service._engine.begin() as conn:
        _make_session(conn, session_id="s2")
        conn.execute(insert(models.chat_messages_table).values(
            id="m1", session_id="s2", role="user", content="hi",
            sequence_no=5, writer_principal="route_user_message",
            created_at=datetime(2026, 4, 30, tzinfo=UTC),
        ))
        base = service._reserve_sequence_range(conn, "s2", count=2)
        assert base == 6


def test_session_write_lock_serializes_sqlite_same_session_sequence_allocation(service):
    """B3 regression: two same-session SQLite writers must not both read
    the same MAX(sequence_no). This test uses the real StaticPool
    in-memory SQLite engine from the shared fixture and two worker
    threads. The sleep happens inside the session write lock to widen
    the race window; without the process-wide per-session lock both
    workers can reserve sequence_no=1 and one insert fails."""
    from concurrent.futures import ThreadPoolExecutor
    import threading
    import time

    from elspeth.web.sessions import models
    from sqlalchemy import insert, select

    barrier = threading.Barrier(2)
    with service._engine.begin() as conn:
        _make_session(conn, session_id="s_sqlite_lock")

    def _writer(index: int) -> int:
        barrier.wait()
        with service._engine.begin() as conn:
            with service._session_write_lock(conn, "s_sqlite_lock"):
                seq = service._reserve_sequence_range(
                    conn, "s_sqlite_lock", count=1
                )
                time.sleep(0.01)
                conn.execute(insert(models.chat_messages_table).values(
                    id=f"m{index}",
                    session_id="s_sqlite_lock",
                    role="user",
                    content=f"message {index}",
                    sequence_no=seq,
                    writer_principal="route_user_message",
                    created_at=datetime(2026, 4, 30, tzinfo=UTC),
                ))
                return seq

    with ThreadPoolExecutor(max_workers=2) as pool:
        seqs = sorted(pool.map(_writer, (1, 2)))

    assert seqs == [1, 2]
    with service._engine.begin() as conn:
        persisted = conn.execute(
            select(models.chat_messages_table.c.sequence_no)
            .where(models.chat_messages_table.c.session_id == "s_sqlite_lock")
            .order_by(models.chat_messages_table.c.sequence_no)
        ).scalars().all()
    assert persisted == [1, 2]
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv/bin/python -m pytest tests/unit/web/sessions/test_persist_compose_turn.py -v -k reserve_sequence
```
Expected: FAIL — helper does not exist.

- [ ] **Step 3: Implement the helper**

In `src/elspeth/web/sessions/service.py`:

```python
def _reserve_sequence_range(
    self, conn: Connection, session_id: str, *, count: int
) -> int:
    """Reserve ``count`` consecutive sequence numbers for ``session_id``.

    PRECONDITION: caller MUST be inside
    ``with self._session_write_lock(conn, session_id):`` in the same
    transaction. That context acquires the PostgreSQL advisory lock or
    the SQLite process-local session lock, making the unilateral
    ``SELECT MAX + 1`` allocation race-free under concurrent writers.

    Inside the same transaction, performs:
        SELECT COALESCE(MAX(sequence_no), 0) FROM chat_messages WHERE session_id = ?
    and returns max+1. The caller writes rows at max+1, max+2, ... max+count.

    The session write lock prevents same-session allocator collisions
    on both PostgreSQL and SQLite. Do not call this helper outside
    the context, even in tests.

    Note: gaps in sequence_no are permitted (transaction rollback after
    reservation leaves the next caller's MAX+1 higher than the first
    successful row's sequence_no). Sequence_no is an ordering key, not a count.

    *Design seam acknowledgement (synthesised review A-F9 / M11).*
    The three-helper protocol (``_acquire_session_advisory_lock`` →
    ``_reserve_sequence_range`` → ``_insert_chat_message``) requires
    callers to invoke them in order. Phase 1 has exactly two callers
    (``persist_compose_turn`` and ``_add_message_sync``); both call
    them correctly. If a Phase 2/3 caller adds a third invocation
    site, consider consolidating into a single
    ``_write_chat_messages_atomic(conn, session_id, rows)`` entry
    point that hides the protocol. Until that third site appears,
    the consolidation is not justified.
    """
    if count < 1:
        raise ValueError(f"count must be >= 1, got {count}")
    # SQLAlchemy 2.x ``select(func.max(...))`` is the project-standard
    # idiom (see existing ``save_composition_state`` at
    # ``service.py:398``); using it here keeps the pattern uniform
    # and gives mypy a typed ``int | None`` instead of the ``Any``
    # that ``text(...)`` plus ``.first()`` produces. Closes
    # synthesised review finding P-L-1 / L15.
    current_max = conn.execute(
        select(func.coalesce(func.max(chat_messages_table.c.sequence_no), 0))
        .where(chat_messages_table.c.session_id == session_id)
    ).scalar_one()
    return int(current_max) + 1
```

- [ ] **Step 4: Run tests to verify pass**

```bash
.venv/bin/python -m pytest tests/unit/web/sessions/test_persist_compose_turn.py -v -k reserve_sequence
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/elspeth/web/sessions/service.py tests/unit/web/sessions/test_persist_compose_turn.py
git commit -m "feat(sessions): add _reserve_sequence_range helper (composer-progress-persistence phase 1)"
```

---

## Task 9: `_insert_chat_message` helper

**Files:**
- Modify: `src/elspeth/web/sessions/service.py`
- Test: `tests/unit/web/sessions/test_persist_compose_turn.py` (extend)

- [ ] **Step 1: Write the failing test**

Add to `tests/unit/web/sessions/test_persist_compose_turn.py`:

```python
def test_insert_chat_message_returns_id_and_persists_row(service):
    from datetime import UTC, datetime
    now = datetime.now(UTC)
    with service._engine.begin() as conn:
        _make_session(conn, session_id="s3")
        msg_id = service._insert_chat_message(
            conn,
            session_id="s3",
            role="assistant",
            content="hello",
            raw_content=None,
            tool_calls=None,
            sequence_no=1,
            writer_principal="compose_loop",
            composition_state_id=None,
            tool_call_id=None,
            parent_assistant_id=None,
            created_at=now,
        )
        assert isinstance(msg_id, str) and len(msg_id) > 0
        rows = conn.execute(text(
            "SELECT id, role, sequence_no, raw_content FROM chat_messages WHERE session_id='s3'"
        )).fetchall()
        assert len(rows) == 1
        assert rows[0].id == msg_id
        assert rows[0].role == "assistant"
        assert rows[0].sequence_no == 1
        assert rows[0].raw_content is None


def test_insert_chat_message_persists_raw_content(service):
    """``raw_content`` is the audit-attribution column for assistant
    messages whose visible content was modified by runtime preflight
    interception. The helper MUST persist it when supplied; if it does
    not, every assistant message after Phase 1 silently loses raw
    attribution data (regression vs the pre-rev-4 ``add_message``)."""
    from datetime import UTC, datetime
    now = datetime.now(UTC)
    with service._engine.begin() as conn:
        _make_session(conn, session_id="s3_raw")
        service._insert_chat_message(
            conn,
            session_id="s3_raw",
            role="assistant",
            content="redacted output",
            raw_content="original LLM output before preflight redaction",
            tool_calls=None,
            sequence_no=1,
            writer_principal="compose_loop",
            composition_state_id=None,
            tool_call_id=None,
            parent_assistant_id=None,
            created_at=now,
        )
        row = conn.execute(text(
            "SELECT content, raw_content FROM chat_messages WHERE session_id='s3_raw'"
        )).first()
        assert row.content == "redacted output"
        assert row.raw_content == "original LLM output before preflight redaction"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
.venv/bin/python -m pytest tests/unit/web/sessions/test_persist_compose_turn.py::test_insert_chat_message_returns_id_and_persists_row -v
```
Expected: FAIL.

- [ ] **Step 3: Implement the helper**

In `src/elspeth/web/sessions/service.py`:

```python
def _insert_chat_message(
    self,
    conn: Connection,
    /,
    *,
    session_id: str,
    role: str,
    content: str,
    raw_content: str | None,
    tool_calls: Any,
    sequence_no: int,
    writer_principal: str,
    composition_state_id: str | None,
    tool_call_id: str | None,
    parent_assistant_id: str | None,
    created_at: datetime,
) -> str:
    """Single-row insert into chat_messages with the supplied fields.

    Caller must be inside ``_session_write_lock`` and have reserved
    ``sequence_no`` from ``_reserve_sequence_range``. ``created_at``
    is supplied by the caller so multi-row inserts in the same
    transaction (``persist_compose_turn``) and same-transaction
    ``sessions.updated_at`` writes (``add_message``) share a single
    timestamp.

    ``raw_content`` is the audit-attribution column for assistant
    messages whose visible ``content`` was rewritten by runtime
    preflight redaction. It MUST be persisted as supplied — silently
    discarding it would regress the pre-rev-4 ``add_message``
    behaviour and create audit-data loss (worse than a crash, per
    CLAUDE.md).
    """
    msg_id = str(uuid.uuid4())
    conn.execute(
        insert(models.chat_messages_table).values(
            id=msg_id,
            session_id=session_id,
            role=role,
            content=content,
            raw_content=raw_content,
            tool_calls=tool_calls,
            sequence_no=sequence_no,
            writer_principal=writer_principal,
            composition_state_id=composition_state_id,
            tool_call_id=tool_call_id,
            parent_assistant_id=parent_assistant_id,
            created_at=created_at,
        )
    )
    return msg_id
```

(Add `import uuid` and `from datetime import datetime` if not already imported, and confirm `from sqlalchemy import insert` and `from . import models` are imported.)

- [ ] **Step 4: Run test to verify it passes**

```bash
.venv/bin/python -m pytest tests/unit/web/sessions/test_persist_compose_turn.py::test_insert_chat_message_returns_id_and_persists_row -v
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/elspeth/web/sessions/service.py tests/unit/web/sessions/test_persist_compose_turn.py
git commit -m "feat(sessions): add _insert_chat_message helper (composer-progress-persistence phase 1)"
```

---

## Task 10: `_insert_composition_state` helper + minimum-touch updates to existing inline inserts

**Why this task's scope is narrower than the earlier plan draft.**

The earlier draft proposed refactoring all existing inline
`composition_states` inserts to use the new helper. Stage 0 grep
showed three such sites, with three different semantics:

- `service.py:~403` (`save_composition_state`): general route-level
  state save; carries its own version-allocation retry loop on
  `IntegrityError`.
- `service.py:~834` (`set_active_state`): same-session state
  revert/pin; uses `derived_from_state_id` lineage.
- `service.py:~1191` (`fork_session`): cross-session state copy at
  fork time.

Forcing a uniform helper across these three structurally-different
sites would either lose the retry loop at site 403 (introducing a
race) or force the helper to grow per-site escape hatches. The
synthesised review panel's H9 finding flagged the original "uniform
refactor + uniform `session_seed`" as papering over real differences.

**Resolved scope (Stage 4 of plan revision, plus B3 + B1 amendments):**

- Add `_insert_composition_state` as a private helper for
  compose-loop and fork-session use. **The helper allocates
  ``composition_states.version`` internally** via
  ``SELECT COALESCE(MAX(version), 0) + 1 FROM composition_states
  WHERE session_id = :sid`` under _session_write_lock (B1 fix —
  see prologue below). ``_StatePayload`` carries no ``version``
  field; the dual-allocator race is closed structurally.
- Use the helper at site `service.py:~1191` (`fork_session`) with
  `provenance="session_fork"`. The fork site supplies no ``version=``
  (it was always ``1`` for a freshly minted session — the helper
  allocates the same value).
- For sites 403 and 834: do NOT restructure through the helper;
  instead make TWO additions to each existing inline insert:
  1. Set `provenance="session_seed"` (broadened semantics — see the
     spec amendment in Task 3).
  2. **Wrap each method's `SELECT MAX` + INSERT region in
     `with self._session_write_lock(conn, sid):` inside the existing
     `engine.begin()` block, BEFORE the SELECT MAX query** (B3 fix —
     see the reachability investigation below). With the lock context
     held, sites 403 and 834 retain their existing inline
     ``SELECT MAX + 1`` allocation BUT it now runs inside the same
     PostgreSQL/SQLite session-write discipline that
     ``_insert_composition_state`` enforces — no caller computes a
     version outside the lock.
- Extract a shared module-level `_enveloped_state_column(...)` helper
  and replace the existing local `_enveloped` helpers in
  `save_composition_state` and `fork_session` with it. The reviewed
  plan's `_insert_composition_state` snippet called `_enveloped(...)`,
  but in live code that helper is local to the two existing methods
  and is not visible from the new private helper.

This keeps the helper focused on a single semantic ("a composition
state derived inside a session-scoped session-write-locked transaction
that does not need its own retry loop") while letting the two
within-session save paths keep their existing structure — and brings
all three writers under the same session-write-lock discipline so their
concurrent execution serialises safely.

**B1 contract change (closes the fabricated-Tier-1-violation vector at the helper-contract level).**

The Phase 1 plan-review synthesis (B1) flagged a related but
distinct failure mode from B3: even with all three writers
(``persist_compose_turn``, ``save_composition_state``,
``set_active_state``) entering the session write lock, the helper's
*signature* still let callers supply a precomputed ``version`` via
``payload.version``. In the Phase 3 compose loop, that version is
read from a ``CompositionState`` snapshot computed BEFORE the sync
worker acquires the lock. Two concurrent compose turns for the same
session can both observe ``state.version = N`` in async land,
serialise into the lock-protected ``_insert_composition_state``
in some order, and both attempt to insert ``version = N + 1`` — the
loser hits ``uq_composition_state_version`` even though the lock
was held the whole time, because the race happened *before* the
lock was acquired.

The locked path's ``IntegrityError`` handler increments
``tool_row_integrity_violation_total`` (SLO threshold = 0) on what
is structurally a contention loss. **Under ELSPETH's auditability
standard, fabricating a Tier-1 audit-integrity alert from a benign
race is evidence-tampering-class harm.**

The fix is to remove ``version`` from ``_StatePayload`` entirely
and have the helper allocate it inside the same transaction that
holds the lock:

```sql
SELECT COALESCE(MAX(version), 0) + 1
  FROM composition_states
 WHERE session_id = :sid
```

The SELECT-MAX-then-INSERT sequence is now atomic against every
other writer for this ``session_id`` because the caller is
contractually required to hold the session write lock before invoking
the helper (PRECONDITION; documented in the helper's docstring
and verified by tests in this task). The dual-allocator race is
foreclosed at the contract boundary, not patched at the call
sites. Closes B1 from the Phase 1 plan-review synthesis.

(B1 is the structural fix; B3 is the call-site-level fix that
brings sites 403 and 834 under the same lock discipline. The two
fixes compose: B3 ensures every writer holds the lock; B1 ensures
no writer can compute a version outside the lock even when it
holds the lock for the INSERT.)

**B3 reachability investigation result (closes the dual-allocator race at the call-site level).**

The Phase 1 plan-review synthesis flagged that `persist_compose_turn`
(`_session_write_lock` protected) and `save_composition_state` / `set_active_state`
(NOT lock-protected, retry-loop-based) both allocate `composition_states.version`
for the same per-session sequence. If reachable concurrently, an
allocator-1 vs allocator-2 interleave produces a real `IntegrityError`
on the locked path → false-positive Tier-1 audit-violation counter
increment + aborted compose-loop turn. **Fabricated Tier-1 violations
are evidence-tampering-class harm under ELSPETH's audit-grade
doctrine.**

**Reachability analysis** (performed against the rev-4 codebase
snapshot — repeat the analysis if the routing surface changes before
Phase 3 wires `persist_compose_turn` in):

- `save_composition_state` is called from seven HTTP-route sites
  (`grep -n "save_composition_state\b" src/elspeth/web/sessions/`):
  routes.py:819, 931, 1133 (compose-error helpers — sequential within
  one compose request), 1742 (compose success path — replaced by
  `persist_compose_turn` in Phase 3), 2145 (recompose success path —
  same), 2556 (fork-time state copy).
- `set_active_state` is called from one site (routes.py:2327, the
  state-revert endpoint).
- The compose endpoint (POST `/sessions/{id}/state/compose`, around
  routes.py:1299) and the state-revert endpoint (POST around
  routes.py:2308) are independent FastAPI handlers. **Nothing in the
  routing layer prevents a user from sending both concurrently for
  the same `session_id`.** FastAPI runs each handler in its own task;
  both can be in-flight simultaneously.
- Therefore in Phase 3, when `persist_compose_turn` is wired into the
  compose endpoint, a concurrent state-revert request (or a fork
  request, or any future endpoint that calls `save_composition_state`)
  produces the dual-allocator race for the same session.

**Conclusion: the race is reachable.** The fix is to enter
`_session_write_lock` in `save_composition_state` and
`set_active_state` so all three writers (`persist_compose_turn`,
`save_composition_state`, `set_active_state`) serialise on the same
per-session lock on both PostgreSQL and SQLite.

**State-intent race (compose vs revert) is separate from allocator
serialization.** The lock makes version allocation safe, but it does
not by itself prove that a compose result was based on the still-current
state. Phase 3's compose endpoint can begin with state A, release the
event loop while the LLM runs, then race a revert/save that makes state B
current. If the compose result then persists under the lock and creates
state C as the newest row, it silently overrides the user's intervening
revert intent. Therefore `persist_compose_turn` must accept
`expected_current_state_id: str | None` and compare it to the current
latest state for the session under `_session_write_lock` before sequence
or version allocation. A mismatch raises `StaleComposeStateError` and
rolls back the compose turn; lock discipline remains necessary but is
not sufficient for intent preservation.

**Effect on the existing retry loops.** Both methods currently wrap
their version-allocation in a 3-attempt retry on `IntegrityError`.
After the lock is held, the SELECT MAX → INSERT sequence is
serialised against every other writer for this `session_id`, so the
retry loop becomes provably unreachable in normal operation. Per
ELSPETH's defensive-programming policy ("don't add error handling
for scenarios that can't happen"), the retry loops SHOULD be
removed once the lock is in place. **Within this task** the loops
are kept as belt-and-suspenders to keep the diff small and bounded;
file an OQ ticket to remove them once the next phase has shaken out
in staging. Mark the kept loops with a `# B3 belt-and-suspenders —
unreachable post-lock; remove in OQ-3-followup` comment so future
readers know the rationale.

**Files:**
- Modify: `src/elspeth/web/sessions/service.py` — add `_insert_composition_state` helper; extract shared `_enveloped_state_column`; refactor only site `~1191` to use the helper; add `provenance="session_seed"` to the inline inserts at sites `~403` and `~834`.
- Test: `tests/unit/web/sessions/test_persist_compose_turn.py` (extend)

- [ ] **Step 1: Write the failing test**

```python
def test_insert_composition_state_returns_id(service):
    from elspeth.web.sessions._persist_payload import _StatePayload
    from elspeth.web.sessions.protocol import CompositionStateData

    with service._engine.begin() as conn:
        # Note: _make_session helper (created in conftest.py during
        # Stage 2 of the plan revision) populates every NOT NULL column
        # on `sessions`. Inline inserts that violated user_id /
        # auth_provider_type / title / updated_at NOT NULL constraints
        # have been removed.
        _make_session(conn, session_id="s4")
        # B1/B3: the helper has a documented precondition — caller
        # MUST be inside the session write-lock context for the same
        # session_id within the same transaction. The context maps to
        # PostgreSQL advisory locks or SQLite per-session process locks.
        with service._session_write_lock(conn, "s4"):
            state_id = service._insert_composition_state(
                conn,
                session_id="s4",
                payload=_StatePayload(
                    # B1: no ``version=``. The helper allocates it.
                    data=CompositionStateData(
                        source={"kind": "tool_response"},
                        nodes=[],
                        edges=[],
                        outputs=[],
                        metadata_={},
                        is_valid=True,
                        validation_errors=None,
                    ),
                    derived_from_state_id="prev_s4_state",
                ),
                provenance="tool_call",
            )
        assert isinstance(state_id, str)
        rows = conn.execute(text(
            "SELECT id, version, provenance, is_valid, derived_from_state_id "
            "FROM composition_states WHERE session_id='s4'"
        )).fetchall()
        assert len(rows) == 1
        # First state for this session: helper allocates COALESCE(MAX,0)+1 = 1.
        assert rows[0].version == 1
        assert rows[0].provenance == "tool_call"
        assert rows[0].is_valid == 1  # SQLite Boolean → INTEGER
        assert rows[0].derived_from_state_id == "prev_s4_state"


def test_insert_composition_state_allocates_contiguous_versions(service):
    """B1 (Phase 1 plan-review synthesis): under the held advisory
    lock, repeated calls to ``_insert_composition_state`` for the same
    session allocate contiguous versions starting at 1. The test runs
    serially within a single transaction. The concurrent same-state
    compose case is exercised on PostgreSQL by Task 16's stale-rejection
    regression."""
    from elspeth.web.sessions._persist_payload import _StatePayload
    from elspeth.web.sessions.protocol import CompositionStateData

    with service._engine.begin() as conn:
        _make_session(conn, session_id="s4_seq")
        ids: list[str] = []
        with service._session_write_lock(conn, "s4_seq"):
            for _ in range(3):
                ids.append(
                    service._insert_composition_state(
                        conn,
                        session_id="s4_seq",
                        payload=_StatePayload(
                            data=CompositionStateData(),
                            derived_from_state_id=None,
                        ),
                        provenance="session_seed",
                    )
                )
        rows = conn.execute(text(
            "SELECT id, version FROM composition_states "
            "WHERE session_id='s4_seq' ORDER BY version"
        )).fetchall()
    assert [r.version for r in rows] == [1, 2, 3]
    assert [r.id for r in rows] == ids


def test_insert_composition_state_versions_are_per_session(service):
    """B1 (Phase 1 plan-review synthesis): the
    ``SELECT COALESCE(MAX(version), 0) + 1`` allocation MUST filter by
    ``session_id``. ``uq_composition_state_version`` is a per-session
    constraint (see Task 1's CREATE TABLE); a global MAX would produce
    versions that are unique cluster-wide but not contiguous within a
    session, breaking the per-session monotonic-version contract every
    other read path assumes (cf. Task 11's
    ``assert states[0].version == 1`` after a fresh session)."""
    from elspeth.web.sessions._persist_payload import _StatePayload
    from elspeth.web.sessions.protocol import CompositionStateData

    with service._engine.begin() as conn:
        _make_session(conn, session_id="s_ver_a")
        _make_session(conn, session_id="s_ver_b")
        with service._session_write_lock(conn, "s_ver_a"):
            for _ in range(5):
                service._insert_composition_state(
                    conn,
                    session_id="s_ver_a",
                    payload=_StatePayload(
                        data=CompositionStateData(),
                        derived_from_state_id=None,
                    ),
                    provenance="session_seed",
                )

    # New transaction; new session. Allocation should restart at 1
    # (per-session), NOT continue at 6 (global).
    with service._engine.begin() as conn:
        with service._session_write_lock(conn, "s_ver_b"):
            service._insert_composition_state(
                conn,
                session_id="s_ver_b",
                payload=_StatePayload(
                    data=CompositionStateData(),
                    derived_from_state_id=None,
                ),
                provenance="session_seed",
            )
        row = conn.execute(text(
            "SELECT version FROM composition_states "
            "WHERE session_id='s_ver_b'"
        )).first()
    assert row.version == 1, (
        f"Per-session version allocation broken: s_ver_b got "
        f"version={row.version}, expected 1. The COALESCE(MAX(version)) "
        "query is missing its WHERE session_id filter."
    )


def test_insert_composition_state_rejects_unknown_provenance(service):
    from sqlalchemy.exc import IntegrityError
    from elspeth.web.sessions._persist_payload import _StatePayload
    from elspeth.web.sessions.protocol import CompositionStateData

    with service._engine.begin() as conn:
        _make_session(conn, session_id="s5")
        # Precondition contract: session write-lock first (see B1/B3 test above).
        with service._session_write_lock(conn, "s5"):
            with pytest.raises(IntegrityError, match="ck_composition_states_provenance"):
                service._insert_composition_state(
                    conn,
                    session_id="s5",
                    payload=_StatePayload(
                        data=CompositionStateData(),
                        derived_from_state_id=None,
                    ),
                    provenance="rogue_value",
                ),
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv/bin/python -m pytest tests/unit/web/sessions/test_persist_compose_turn.py -v -k insert_composition_state
```
Expected: FAIL.

- [ ] **Step 3: Implement the helper and refactor existing inline inserts**

Add the shared envelope helper at module scope, then add
`_insert_composition_state` to `SessionServiceImpl`:

```python
def _enveloped_state_column(value: Any) -> Any:
    """Return the JSON envelope stored by composition_states JSON columns.

    Existing `save_composition_state` and `fork_session` each carried a
    local `_enveloped` helper. `_insert_composition_state` is module/class
    scope, so the envelope rule must be extracted before the helper can
    call it. Do not duplicate the helper back into individual methods.
    """
    raw = deep_thaw(value)
    if raw is None:
        return None
    return {"_version": 1, "data": raw}


def _insert_composition_state(
    self,
    conn: Connection,
    *,
    session_id: str,
    payload: _StatePayload,
    provenance: str,
    created_at: datetime | None = None,
) -> str:
    """Single-row insert into composition_states with per-session
    version allocation under _session_write_lock.

    PRECONDITION: caller MUST be inside
    ``with self._session_write_lock(conn, session_id):`` in the same
    transaction. The context is what makes the
    ``SELECT COALESCE(MAX(version), 0) + 1 FROM composition_states
    WHERE session_id = :sid`` allocation race-free under concurrent
    writers — without it, two callers could both observe MAX = N,
    both pick N+1, and the loser's INSERT would hit
    ``uq_composition_state_version``. The locked-path
    ``IntegrityError`` handler classifies that as a Tier-1
    audit-integrity violation, fabricating a Tier-1 alert from a
    benign contention loss. **Under ELSPETH's auditability standard
    fabricated Tier-1 violations are evidence-tampering-class harm:
    the audit trail asserts a violation that did not occur.** The
    session write lock makes the SELECT-MAX-then-INSERT sequence
    atomic against every other writer for this ``session_id`` on both
    PostgreSQL and SQLite, so the fabrication path is structurally
    unreachable. Closes B1/B3 from the Phase 1 plan-review synthesis.
    Also mirrors the precondition contract on ``_reserve_sequence_range``.

    Version allocation is per-session: the COALESCE query filters by
    ``session_id`` because ``uq_composition_state_version`` is a
    per-session constraint (see Task 1's CREATE TABLE). A global MAX
    would silently break the per-session monotonic-version contract
    every read path assumes.

    This helper does NOT contain a retry loop. The lock + atomic
    SELECT-INSERT makes one a defensive-programming anti-pattern.

    Writes the real per-column schema (source/nodes/edges/outputs/
    metadata_/is_valid/validation_errors/derived_from_state_id), using
    the shared ``_enveloped_state_column(...)`` and ``deep_thaw(...)``
    patterns the existing inline inserts use today.

    The ``provenance`` argument must satisfy the
    ``ck_composition_states_provenance`` CHECK constraint added in
    Task 3; passing an unknown value raises ``IntegrityError``.

    The ``created_at`` argument is optional. When ``None`` (the
    default), the helper stamps ``datetime.now(UTC)`` at
    insert time. Callers that need cross-table timestamp consistency
    within a single transaction (e.g. ``fork_session`` pre-computes
    ``now`` at the top of its sync block and reuses it across the
    ``sessions``, ``composition_states``, and ``chat_messages``
    inserts so all rows share one wall-clock instant) MUST pass an
    explicit ``created_at``. Earlier B1 framing (helper hardcoding
    ``now()`` silently changed fork timestamp semantics) is preserved.
    """
    # B1: allocate version under _session_write_lock. The
    # SELECT-MAX-then-INSERT sequence is atomic against every other
    # writer for this session because the caller is required to be
    # inside ``_session_write_lock(conn, session_id)`` for the full
    # transaction (see PRECONDITION above). The COALESCE pins the
    # first state to version 1; the WHERE clause makes the allocation
    # per-session, matching ``uq_composition_state_version``'s scope.
    next_version = conn.execute(
        select(
            func.coalesce(
                func.max(models.composition_states_table.c.version), 0
            )
            + 1
        ).where(
            models.composition_states_table.c.session_id == session_id
        )
    ).scalar_one()
    state_id = str(uuid.uuid4())
    conn.execute(
        insert(models.composition_states_table).values(
            id=state_id,
            session_id=session_id,
            version=int(next_version),
            source=_enveloped_state_column(payload.data.source),
            nodes=_enveloped_state_column(payload.data.nodes),
            edges=_enveloped_state_column(payload.data.edges),
            outputs=_enveloped_state_column(payload.data.outputs),
            metadata_=_enveloped_state_column(payload.data.metadata_),
            is_valid=payload.data.is_valid,
            validation_errors=deep_thaw(payload.data.validation_errors),
            derived_from_state_id=payload.derived_from_state_id,
            provenance=provenance,
            created_at=created_at if created_at is not None else datetime.now(UTC),
        )
    )
    return state_id
```

(Add `Any` to the typing imports, and add `from elspeth.contracts.freeze import deep_thaw` and `from elspeth.web.sessions._persist_payload import _StatePayload` to the imports if not already present. ``select`` and ``func`` are already imported at the module top — they are used by ``save_composition_state`` and ``_reserve_sequence_range`` for the same SQLAlchemy 2.x SELECT-MAX idiom; no new import is required for the B1 version-allocation query. Replace the existing method-local `_enveloped` helpers in `save_composition_state` and `fork_session` with `_enveloped_state_column` in this same task.)

Then make the per-site updates. There are **three** existing inline
`composition_states` inserts in `service.py`; locate them via
`grep -n 'insert(composition_states_table\|insert(models.composition_states_table)' src/elspeth/web/sessions/service.py`
(line numbers may have drifted from the snapshot below).

**Site-by-site mapping:**

| Site | Method | Treatment | Provenance |
|---|---|---|---|
| `service.py:~403` | `save_composition_state` (general route-level state save) | KEEP inline. Three additions: (1) add `provenance="session_seed",` to the existing `.values(...)` call; (2) wrap the existing `SELECT MAX(version)` + INSERT retry body in `with self._session_write_lock(conn, sid):` as the FIRST operation inside the `engine.begin()` block at `_try_insert_state` (line ~396); (3) replace the local `_enveloped` helper with shared `_enveloped_state_column`. The context makes the SELECT-then-INSERT sequence atomic against every other writer for this `session_id` — closing B3 from the Phase 1 plan-review synthesis. The retry loop is kept as belt-and-suspenders (see B3 prologue above). | `"session_seed"` (broadened semantics — spec §4.1.2 amendment in Task 3) |
| `service.py:~834` | `set_active_state` (same-session state revert/pin) | KEEP inline. Two additions: (1) add `provenance="session_seed",` to the `.values(...)` call; (2) wrap the prior-row SELECT, `SELECT MAX(version)`, and INSERT in `with self._session_write_lock(conn, sid):` inside the `engine.begin()` block at `_try_insert_revert` (line ~794). The `derived_from_state_id` is already populated correctly. The retry loop is kept as belt-and-suspenders (see B3 prologue above). | `"session_seed"` (same broadened semantics) |
| `service.py:~1191` | `fork_session` (cross-session state copy at fork) | REFACTOR to call `_insert_composition_state`. Build a `_StatePayload` from the source state's fields and call the helper with `provenance="session_fork"` and `created_at=now`. Note: §14.6's fork sweep already requires entering `_session_write_lock` for the new session_id BEFORE this state insert (see §14.6 Step 1.5). The fork is a single-shot insert with no retry loop, which matches the helper's contract. | `"session_fork"` (new enum value — see Task 3) |

The site-1191 refactor (illustrative — adapt to the surrounding code shape):

```python
# BEFORE (~service.py:1190-1205)
state_version = 1
conn.execute(
    insert(composition_states_table).values(
        id=copied_state_id_str,
        session_id=new_session_id_str,
        version=1,
        source=_enveloped(source_state_record.source),
        ...
        derived_from_state_id=None,
        created_at=now,
    )
)

# AFTER
# The new session_id was minted seconds ago; no other writer can know
# it yet, so the lock is technically uncontended. We acquire it anyway
# because the helper's PRECONDITION contract requires it (see Task 10's
# B1/B3 docstring) and honouring the contract uniformly is cheaper than
# reasoning about exceptions. §14.6's fork sweep wraps the state and
# message inserts in ``with self._session_write_lock(conn,
# new_session_id_str):`` BEFORE this state insert; if the order is
# changed in a future refactor, restore it here.
with self._session_write_lock(conn, new_session_id_str):
    self._insert_composition_state(
        conn,
        session_id=new_session_id_str,
        payload=_StatePayload(
            # B1 (Phase 1 plan-review synthesis): no ``version=``. The
            # helper allocates COALESCE(MAX(version),0)+1 = 1 (this is the
            # first state in a freshly created session) under the held
            # session write lock. Removing the kwarg makes the
            # dual-allocator race structurally impossible; previously a
            # caller could supply a stale version computed outside the lock.
            data=CompositionStateData(
                source=source_state_record.source,
                nodes=source_state_record.nodes,
                edges=source_state_record.edges,
                outputs=source_state_record.outputs,
                metadata_=source_state_record.metadata_,
                is_valid=source_state_record.is_valid,
                validation_errors=source_state_record.validation_errors,
            ),
            derived_from_state_id=None,
        ),
        provenance="session_fork",
        created_at=now,  # cross-table timestamp consistency — pre-computed at line ~1104
    )
# The pre-rev-4 ``state_version`` local was used by downstream code
# that needed to know the version of the just-inserted state. With
# version allocated by the helper, that information is no longer
# available at the call site. If a downstream consumer needs it,
# query ``composition_states.version`` for ``copied_state_id_str``
# inside the same transaction; for a fresh session the value is
# always 1.
```

(The helper internally rewraps the per-column data via
`_enveloped_state_column(...)` and `deep_thaw(...)`, matching the surrounding
inline-insert behaviour. The explicit ``created_at=now`` parameter
preserves the cross-table timestamp invariant that the original
inline insert relied on — see B1 in the Phase 1 plan-review JSON
for context.)

The other two sites (`~403` and `~834`) stay inline, but they are not
"provenance only": they also enter `_session_write_lock` around their
SELECT-MAX/INSERT regions, and `save_composition_state` must use the
shared `_enveloped_state_column` helper instead of keeping a private
method-local duplicate.

- [ ] **Step 4: Run tests to verify pass**

```bash
.venv/bin/python -m pytest tests/unit/web/sessions/test_persist_compose_turn.py -v -k insert_composition_state
.venv/bin/python -m pytest tests/unit/web/sessions/ -v
```
Expected: PASS for the new tests AND every previously-passing sessions test continues to pass (the refactor must not break existing behaviour).

- [ ] **Step 5: Commit**

```bash
git add src/elspeth/web/sessions/service.py tests/unit/web/sessions/test_persist_compose_turn.py
git commit -m "feat(sessions): add _insert_composition_state helper and extend session write lock to state writers (B3 dual-allocator race fix; composer-progress-persistence phase 1)"
```

---

## Task 11: `persist_compose_turn` happy path

**Files:**
- Modify: `src/elspeth/web/sessions/service.py`
- Test: `tests/unit/web/sessions/test_persist_compose_turn.py` (extend)

- [ ] **Step 1: Write the failing test**

```python
def test_persist_compose_turn_happy_path(service):
    from elspeth.web.sessions._persist_payload import (
        _RedactedToolRow, _StatePayload,
    )
    from elspeth.web.sessions.protocol import CompositionStateData
    with service._engine.begin() as conn:
        _make_session(conn, session_id="s6")

    outcome = service.persist_compose_turn(
        session_id="s6",
        assistant_content="ok",
        redacted_assistant_tool_calls=({"id": "tc_1", "function": {"name": "set_source"}},),
        redacted_tool_rows=(
            _RedactedToolRow(
                tool_call_id="tc_1",
                content='{"ok": true}',
                # B1 (Phase 1 plan-review synthesis): no ``version=``.
                # ``_insert_composition_state`` allocates it under the
                # held session write lock; the assertion below pins the
                # allocated value to 1 (first state in this session).
                composition_state_payload=_StatePayload(
                    data=CompositionStateData(),
                    derived_from_state_id=None,
                ),
            ),
        ),
        parent_composition_state_id=None,
        expected_current_state_id=None,
        writer_principal="compose_loop",
        plugin_crash_pending=False,
    )

    # On the success path, _AuditOutcome carries the new
    # assistant_id and unwind_audit_failed=False. The old
    # tier1_violation field was removed in Stage 4 of the plan
    # revision (Tier-1 failures now raise AuditIntegrityError
    # directly — see Task 13).
    assert outcome.unwind_audit_failed is False
    assert outcome.assistant_id is not None

    with service._engine.begin() as conn:
        rows = conn.execute(text(
            "SELECT role, sequence_no, tool_call_id "
            "FROM chat_messages WHERE session_id='s6' ORDER BY sequence_no"
        )).fetchall()
        assert [r.role for r in rows] == ["assistant", "tool"]
        assert rows[0].sequence_no == 1
        assert rows[1].sequence_no == 2
        assert rows[1].tool_call_id == "tc_1"

        states = conn.execute(text(
            "SELECT version, provenance FROM composition_states WHERE session_id='s6'"
        )).fetchall()
        assert len(states) == 1
        assert states[0].version == 1
        assert states[0].provenance == "tool_call"


def test_persist_compose_turn_zero_tool_rows(service):
    """W10a (Phase 1 plan-review synthesis): a turn with
    ``redacted_tool_rows=()`` and ``redacted_assistant_tool_calls=()``
    is a valid and reachable shape — the assistant produced text but
    chose not to call any tools. Spec §5.2 explicitly allows this. The
    primitive MUST commit cleanly: the assistant row is persisted, no
    tool rows are inserted, and no ``composition_states`` rows are
    created (because the empty tool-row tuple has no
    ``composition_state_payload`` to write).

    The zero-row case is not exercised by ``happy_path`` (which always
    includes one ``_RedactedToolRow``), so without this regression the
    next caller migrating an assistant-only call site (Phase 3) would
    discover an off-by-one or empty-tuple bug at integration time
    rather than at the primitive's own unit boundary.
    """
    with service._engine.begin() as conn:
        _make_session(conn, session_id="s_zero")
    outcome = service.persist_compose_turn(
        session_id="s_zero",
        assistant_content="text only",
        redacted_assistant_tool_calls=(),
        redacted_tool_rows=(),
        parent_composition_state_id=None,
        expected_current_state_id=None,
        writer_principal="compose_loop",
        plugin_crash_pending=False,
    )
    assert outcome.assistant_id is not None
    assert outcome.unwind_audit_failed is False
    with service._engine.begin() as conn:
        roles = [
            r.role for r in conn.execute(text(
                "SELECT role FROM chat_messages WHERE session_id='s_zero'"
            )).fetchall()
        ]
        assert roles == ["assistant"]
        states = conn.execute(text(
            "SELECT id FROM composition_states WHERE session_id='s_zero'"
        )).fetchall()
        assert states == []


def test_persist_compose_turn_persists_raw_content(service):
    """B2 (Phase 1 plan-review synthesis): ``persist_compose_turn`` must
    plumb the optional ``raw_content`` argument to the assistant row
    verbatim. ``raw_content`` is the audit-attribution column that
    captures the original LLM output BEFORE preflight redaction
    rewrote ``content``. Routes 1749 and 2152 (formerly 1542/1945
    pre-rev-4) already passed ``raw_content=result.raw_assistant_content`` to
    ``add_message``; Phase 3 migrates those call sites to
    ``persist_compose_turn``, so the primitive must accept and persist
    the column today (Phase 1) — not later (Phase 3, which is
    explicitly "loop only, no new primitives").

    Pre-B2 ``persist_compose_turn`` hardcoded ``raw_content=None`` at
    the assistant-row insert; this test would have failed by reading
    ``None`` from ``chat_messages.raw_content`` instead of the supplied
    string. Tool rows still get ``raw_content=None`` regardless —
    redaction-attribution applies only to LLM-authored content.
    """
    from elspeth.web.sessions._persist_payload import _RedactedToolRow

    with service._engine.begin() as conn:
        _make_session(conn, session_id="s6_raw")

    outcome = service.persist_compose_turn(
        session_id="s6_raw",
        assistant_content="ok (redacted)",
        raw_content="original LLM output before preflight redaction",
        redacted_assistant_tool_calls=({"id": "tc_1", "function": {"name": "f"}},),
        redacted_tool_rows=(
            _RedactedToolRow(tool_call_id="tc_1", content="{}", composition_state_payload=None),
        ),
        parent_composition_state_id=None,
        expected_current_state_id=None,
        writer_principal="compose_loop",
        plugin_crash_pending=False,
    )
    assert outcome.assistant_id is not None
    assert outcome.unwind_audit_failed is False

    with service._engine.begin() as conn:
        rows = conn.execute(text(
            "SELECT role, content, raw_content FROM chat_messages "
            "WHERE session_id='s6_raw' ORDER BY sequence_no"
        )).fetchall()
        # Assistant row carries both visible content (post-redaction)
        # and raw_content (pre-redaction); tool row has raw_content=None.
        assert rows[0].role == "assistant"
        assert rows[0].content == "ok (redacted)"
        assert rows[0].raw_content == "original LLM output before preflight redaction"
        assert rows[1].role == "tool"
        assert rows[1].raw_content is None


def test_persist_compose_turn_rejects_cross_session_parent_state(service):
    """B5 (Phase 1 plan-review synthesis): when ``parent_composition_state_id``
    is supplied and points to a state that belongs to a DIFFERENT session,
    the call MUST raise ``RuntimeError`` with the precise diagnostic
    produced by ``_assert_state_in_session`` — not a generic FK error.

    The composite FK ``fk_chat_messages_composition_state_session`` would
    eventually catch the mismatch at INSERT time, but the ELSPETH
    offensive-programming policy requires a named pre-check at the
    service boundary so the diagnostic identifies the caller, the state,
    and the session mismatch (CLAUDE.md "Defensive Programming:
    Forbidden" — "Proactively detect invalid states and throw meaningful
    exceptions").

    This test would have failed pre-B5 by either crashing with an opaque
    ``IntegrityError`` (if the FK fired) or by silently inserting a row
    that the FK validated only via column equality, producing an
    audit-trail that lies about which session authored the row.
    """
    from elspeth.web.sessions._persist_payload import _StatePayload
    from elspeth.web.sessions.protocol import CompositionStateData

    # Set up two sessions; insert a composition_state into session A.
    with service._engine.begin() as conn:
        _make_session(conn, session_id="s_A")
        _make_session(conn, session_id="s_B")
        with service._session_write_lock(conn, "s_A"):
            state_a_id = service._insert_composition_state(
                conn,
                session_id="s_A",
                # B1: no ``version=`` — helper allocates under the lock.
                payload=_StatePayload(
                    data=CompositionStateData(),
                    derived_from_state_id=None,
                ),
                provenance="session_seed",
            )

    # Now try to persist a turn on session B that references session A's state.
    # The guard MUST fire BEFORE the FK does and produce the precise
    # _assert_state_in_session diagnostic.
    with pytest.raises(
        RuntimeError,
        match=r"persist_compose_turn: composition_state_id=.*belongs to session "
              r"'s_A', not 's_B'.*cross-session reference is a contract violation",
    ):
        service.persist_compose_turn(
            session_id="s_B",
            assistant_content="should fail",
            redacted_assistant_tool_calls=(),
            redacted_tool_rows=(),
            parent_composition_state_id=state_a_id,
            expected_current_state_id=None,
            writer_principal="compose_loop",
            plugin_crash_pending=False,
        )

    # Post-condition: no chat_messages rows were inserted on s_B.
    # The guard fires INSIDE the transaction, so the engine.begin() context
    # rolls back any partial work; verify the row count is zero.
    with service._engine.begin() as conn:
        b_count = conn.execute(text(
            "SELECT COUNT(*) FROM chat_messages WHERE session_id='s_B'"
        )).scalar()
        assert b_count == 0, (
            f"persist_compose_turn rolled back incorrectly; s_B has "
            f"{b_count} chat rows after a guard-rejected call"
        )


def test_persist_compose_turn_accepts_valid_same_session_parent_state(service):
    """B5 happy path: when ``parent_composition_state_id`` references a
    state that belongs to the SAME session, the guard passes silently and
    the assistant row is correctly stamped with that
    ``composition_state_id``. Closes a coverage gap noted in the quality
    reviewer's W-3 (no test exercised valid non-None parent state)."""
    from elspeth.web.sessions._persist_payload import _StatePayload
    from elspeth.web.sessions.protocol import CompositionStateData

    with service._engine.begin() as conn:
        _make_session(conn, session_id="s_C")
        with service._session_write_lock(conn, "s_C"):
            state_c_id = service._insert_composition_state(
                conn,
                session_id="s_C",
                # B1: no ``version=`` — helper allocates under the lock.
                payload=_StatePayload(
                    data=CompositionStateData(),
                    derived_from_state_id=None,
                ),
                provenance="session_seed",
            )

    outcome = service.persist_compose_turn(
        session_id="s_C",
        assistant_content="ok",
        redacted_assistant_tool_calls=(),
        redacted_tool_rows=(),
        parent_composition_state_id=state_c_id,
        expected_current_state_id=state_c_id,
        writer_principal="compose_loop",
        plugin_crash_pending=False,
    )

    assert outcome.unwind_audit_failed is False
    assert outcome.assistant_id is not None

    with service._engine.begin() as conn:
        assistant_row = conn.execute(text(
            "SELECT composition_state_id FROM chat_messages "
            "WHERE session_id='s_C' AND role='assistant'"
        )).fetchone()
        assert assistant_row is not None
        assert assistant_row.composition_state_id == state_c_id


def test_persist_compose_turn_rejects_stale_expected_current_state(service):
    """A compose turn may not persist if the session's current state
    changed while the LLM call was in flight.

    The session-write lock serializes the DB mutation, but the intent
    check is what prevents a compose based on state A from becoming the
    newest state after a concurrent revert/save already made state B
    current. Closes the compose-vs-revert race identified in plan review.
    """
    from elspeth.web.sessions._persist_payload import _StatePayload
    from elspeth.web.sessions.protocol import CompositionStateData
    from elspeth.web.sessions.service import StaleComposeStateError

    with service._engine.begin() as conn:
        _make_session(conn, session_id="s_stale")
        with service._session_write_lock(conn, "s_stale"):
            stale_state_id = service._insert_composition_state(
                conn,
                session_id="s_stale",
                payload=_StatePayload(
                    data=CompositionStateData(),
                    derived_from_state_id=None,
                ),
                provenance="session_seed",
            )
            current_state_id = service._insert_composition_state(
                conn,
                session_id="s_stale",
                payload=_StatePayload(
                    data=CompositionStateData(),
                    derived_from_state_id=stale_state_id,
                ),
                provenance="session_seed",
            )

    with pytest.raises(
        StaleComposeStateError,
        match=r"current composition state changed.*expected=.*actual=",
    ):
        service.persist_compose_turn(
            session_id="s_stale",
            assistant_content="stale",
            redacted_assistant_tool_calls=(),
            redacted_tool_rows=(),
            parent_composition_state_id=stale_state_id,
            expected_current_state_id=stale_state_id,
            writer_principal="compose_loop",
            plugin_crash_pending=False,
        )

    with service._engine.begin() as conn:
        rows = conn.execute(text(
            "SELECT role FROM chat_messages WHERE session_id='s_stale'"
        )).fetchall()
        latest = conn.execute(text(
            "SELECT id FROM composition_states WHERE session_id='s_stale' "
            "ORDER BY version DESC LIMIT 1"
        )).scalar_one()
    assert rows == []
    assert latest == current_state_id


def test_persist_compose_turn_accepts_matching_expected_current_state(service):
    from elspeth.web.sessions._persist_payload import _StatePayload
    from elspeth.web.sessions.protocol import CompositionStateData

    with service._engine.begin() as conn:
        _make_session(conn, session_id="s_current_ok")
        with service._session_write_lock(conn, "s_current_ok"):
            current_state_id = service._insert_composition_state(
                conn,
                session_id="s_current_ok",
                payload=_StatePayload(
                    data=CompositionStateData(),
                    derived_from_state_id=None,
                ),
                provenance="session_seed",
            )

    outcome = service.persist_compose_turn(
        session_id="s_current_ok",
        assistant_content="ok",
        redacted_assistant_tool_calls=(),
        redacted_tool_rows=(),
        parent_composition_state_id=None,
        expected_current_state_id=current_state_id,
        writer_principal="compose_loop",
        plugin_crash_pending=False,
    )
    assert outcome.assistant_id is not None
    assert outcome.unwind_audit_failed is False


@pytest.mark.asyncio
async def test_persist_compose_turn_refuses_async_invocation(service):
    """Calling ``persist_compose_turn`` directly from a coroutine
    must raise RuntimeError. Production callers (Phase 3 compose
    loop) use ``await service.persist_compose_turn_async(...)``, which
    dispatches to a worker thread; the body's
    synchronous SQLAlchemy transaction would otherwise block the
    event loop.

    Closes synthesised review finding SA-7 / M1 (async-loop guard
    is convention-only without a runtime check)."""
    with service._engine.begin() as conn:
        _make_session(conn, session_id="s_async_guard")

    with pytest.raises(RuntimeError, match="must be dispatched via"):
        # Calling the SYNC method from inside an async test function
        # — there IS a running loop in the calling thread, which is
        # exactly the misuse the guard exists to detect.
        service.persist_compose_turn(
            session_id="s_async_guard",
            assistant_content="",
            redacted_assistant_tool_calls=(),
            redacted_tool_rows=(),
            parent_composition_state_id=None,
            expected_current_state_id=None,
            writer_principal="compose_loop",
            plugin_crash_pending=False,
        )


@pytest.mark.asyncio
async def test_persist_compose_turn_async_protocol_dispatch_succeeds_from_async(service):
    """Companion to the async-guard test: production callers use the
    protocol-public async dispatcher, not the concrete sync primitive
    and not a concrete ``_run_sync`` bridge. The dispatcher runs the sync
    primitive in a worker thread (no running loop in that thread), so the
    guard passes while routes keep depending on SessionServiceProtocol."""
    from elspeth.web.sessions._persist_payload import _RedactedToolRow

    with service._engine.begin() as conn:
        _make_session(conn, session_id="s_run_sync")

    outcome = await service.persist_compose_turn_async(
        session_id="s_run_sync",
        assistant_content="ok",
        redacted_assistant_tool_calls=(
            {"id": "tc_run_sync", "function": {"name": "f"}},
        ),
        redacted_tool_rows=(_RedactedToolRow("tc_run_sync", "{}", None),),
        parent_composition_state_id=None,
        expected_current_state_id=None,
        writer_principal="compose_loop",
        plugin_crash_pending=False,
    )
    assert outcome.assistant_id is not None
    assert outcome.unwind_audit_failed is False
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv/bin/python -m pytest tests/unit/web/sessions/test_persist_compose_turn.py -v -k "happy_path or zero_tool_rows or persists_raw_content or rejects_cross_session_parent_state or accepts_valid_same_session_parent_state or rejects_stale_expected_current_state or accepts_matching_expected_current_state or refuses_async_invocation or async_protocol_dispatch_succeeds_from_async"
```
Expected: FAIL — `persist_compose_turn` does not yet exist (or, after Step 3, a partial implementation is missing the `raw_content` plumbing that
`test_persist_compose_turn_persists_raw_content` asserts).

- [ ] **Step 3: Implement `persist_compose_turn` (success path only)**

In `src/elspeth/web/sessions/service.py`, add the method (full body in spec §5.2.2; success-path implementation here, error-path tasks later):

```python
class StaleComposeStateError(RuntimeError):
    """Compose result was based on a no-longer-current composition state."""


def persist_compose_turn(
    self,
    *,
    session_id: str,
    assistant_content: str,
    raw_content: str | None = None,
    redacted_assistant_tool_calls: tuple[Mapping[str, Any], ...],
    redacted_tool_rows: tuple[_RedactedToolRow, ...],
    parent_composition_state_id: str | None,
    expected_current_state_id: str | None,
    writer_principal: str,
    plugin_crash_pending: bool,
) -> _AuditOutcome:
    """Synchronous, single-transaction persistence of one compose turn.

    Spec §5.2.2. Concrete sync primitive. Production async callers MUST
    invoke ``await self.persist_compose_turn_async(...)`` through
    ``SessionServiceProtocol``; that dispatcher uses ``_run_sync`` under
    the hood. Calling this sync primitive directly from async land would
    block the event loop because the body opens a synchronous SQLAlchemy
    transaction.

    The guard below uses ``asyncio.get_running_loop()`` to detect
    misuse: if there is a running loop in the calling thread, we are
    in async land and MUST refuse. ``RuntimeError`` is the canonical
    "you called the wrong API" signal — the call site is a bug, not
    a recoverable user error. Closes synthesised review finding
    SA-7 / M1.
    """
    import asyncio
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        # No running loop in this thread — we are in a worker thread
        # or pure sync test context. Proceed.
        pass
    else:
        raise RuntimeError(
            "persist_compose_turn must be dispatched via "
            "await self.persist_compose_turn_async(...) — "
            "calling it directly from a coroutine blocks the event "
            "loop on synchronous DB I/O."
        )

    now = self._now()
    with self._engine.begin() as conn:
        with self._session_write_lock(conn, session_id):
            # B5 (Phase 1 plan-review synthesis): if a parent composition
            # state is supplied, it MUST belong to this session. The composite
            # FK ``fk_chat_messages_composition_state_session`` would catch a
            # cross-session reference at INSERT time, but only with a generic
            # FK error AFTER the row attempt. The offensive-programming policy
            # (CLAUDE.md "Defensive Programming: Forbidden") requires a precise
            # named pre-check that produces a diagnostic identifying the caller,
            # the state, and the mismatched session. ``_assert_state_in_session``
            # is the canonical guard (already called by ``add_message``); using
            # it here brings ``persist_compose_turn`` to the same offensive
            # standard. Mirrors the §14.6 fork-slice parent-assistant guard.
            if parent_composition_state_id is not None:
                _assert_state_in_session(
                    conn,
                    state_id=parent_composition_state_id,
                    expected_session_id=session_id,
                    caller="persist_compose_turn",
                )

            current_state_id = conn.execute(
                select(models.composition_states_table.c.id)
                .where(models.composition_states_table.c.session_id == session_id)
                .order_by(models.composition_states_table.c.version.desc())
                .limit(1)
            ).scalar_one_or_none()
            if current_state_id != expected_current_state_id:
                raise StaleComposeStateError(
                    "persist_compose_turn: current composition state changed "
                    f"for session_id={session_id!r}; "
                    f"expected={expected_current_state_id!r}, "
                    f"actual={current_state_id!r}. Refusing to persist a "
                    "compose result based on a stale state."
                )

            base_seq = self._reserve_sequence_range(
                conn, session_id, count=1 + len(redacted_tool_rows)
            )

            assistant_id = self._insert_chat_message(
                conn,
                session_id=session_id,
                role="assistant",
                content=assistant_content,
                # B2 (Phase 1 plan-review synthesis): ``raw_content`` is the
                # audit-attribution column for assistant messages whose
                # ``content`` was rewritten by runtime preflight redaction.
                # Pre-B2 the parameter was hardcoded ``None`` here with a
                # "Phase 3 plumbs through then" comment, but Phase 3 is
                # described in its own plan as "loop only, no new
                # primitives" — Phase 3 wires the call site, it does not
                # extend the signature. Routes 1749 and 2152 already pass
                # ``raw_content=result.raw_assistant_content`` to
                # ``add_message`` today; persist_compose_turn must accept
                # the same column from Phase 3 onward, so the parameter is
                # introduced in Phase 1 with a default of ``None`` (which
                # preserves existing Phase 1 happy-path test behaviour
                # unchanged) and Phase 3 supplies the LLM-response value.
                raw_content=raw_content,
                # ``deep_thaw`` recursively converts any MappingProxyType /
                # tuple inputs to JSON-serializable dict / list forms; this
                # is the same pattern ``save_composition_state`` uses for
                # frozen ``validation_errors`` (service.py:413). Plain
                # ``list(...)`` would leave MappingProxyType inner values
                # unchanged, which json.dumps then rejects with TypeError.
                # Closes synthesised review finding P-L-4 / L18.
                tool_calls=deep_thaw(redacted_assistant_tool_calls) if redacted_assistant_tool_calls else None,
                sequence_no=base_seq,
                writer_principal=writer_principal,
                composition_state_id=parent_composition_state_id,
                tool_call_id=None,
                parent_assistant_id=None,
                created_at=now,
            )

            for offset, tool_row in enumerate(redacted_tool_rows, start=1):
                state_id: str | None = None
                if tool_row.composition_state_payload is not None:
                    state_id = self._insert_composition_state(
                        conn,
                        session_id=session_id,
                        payload=tool_row.composition_state_payload,
                        provenance="tool_call",
                    )
                self._insert_chat_message(
                    conn,
                    session_id=session_id,
                    role="tool",
                    content=tool_row.content,
                    raw_content=None,
                    tool_calls=None,
                    sequence_no=base_seq + offset,
                    writer_principal=writer_principal,
                    composition_state_id=state_id,
                    tool_call_id=tool_row.tool_call_id,
                    parent_assistant_id=assistant_id,
                    created_at=now,
                )

        return _AuditOutcome(
            assistant_id=assistant_id,
            unwind_audit_failed=False,
        )
```

- [ ] **Step 3b: Expose only the async dispatcher on `SessionServiceProtocol`**

In `src/elspeth/web/sessions/protocol.py`, add the protocol method in
the same task that introduces `SessionServiceImpl.persist_compose_turn`.
This keeps the overview's public contract true for Phase 1 and prevents
Phase 3 from reaching into the concrete service type or its `_run_sync`
bridge. The sync `SessionServiceImpl.persist_compose_turn` remains
concrete-only and guarded against direct async-loop use.

Avoid a runtime circular import: Task 6's `_persist_payload.py` imports
`CompositionStateData` from `protocol.py`, so `protocol.py` must import
the payload dataclasses only for static typing.

```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from elspeth.web.sessions._persist_payload import _AuditOutcome, _RedactedToolRow


class SessionServiceProtocol(Protocol):
    # ... existing methods ...

    async def persist_compose_turn_async(
        self,
        *,
        session_id: str,
        assistant_content: str,
        raw_content: str | None = None,
        redacted_assistant_tool_calls: tuple[Mapping[str, Any], ...],
        redacted_tool_rows: tuple[_RedactedToolRow, ...],
        parent_composition_state_id: str | None,
        expected_current_state_id: str | None,
        writer_principal: str,
        plugin_crash_pending: bool,
    ) -> _AuditOutcome: ...
```

In `SessionServiceImpl`, add the matching async dispatcher with the same
signature:

```python
async def persist_compose_turn_async(
    self,
    *,
    session_id: str,
    assistant_content: str,
    raw_content: str | None = None,
    redacted_assistant_tool_calls: tuple[Mapping[str, Any], ...],
    redacted_tool_rows: tuple[_RedactedToolRow, ...],
    parent_composition_state_id: str | None,
    expected_current_state_id: str | None,
    writer_principal: str,
    plugin_crash_pending: bool,
) -> _AuditOutcome:
    return await self._run_sync(
        self.persist_compose_turn,
        session_id=session_id,
        assistant_content=assistant_content,
        raw_content=raw_content,
        redacted_assistant_tool_calls=redacted_assistant_tool_calls,
        redacted_tool_rows=redacted_tool_rows,
        parent_composition_state_id=parent_composition_state_id,
        expected_current_state_id=expected_current_state_id,
        writer_principal=writer_principal,
        plugin_crash_pending=plugin_crash_pending,
    )
```

If `from __future__ import annotations` is ever removed from this file,
the annotation strategy must be revisited before landing the change.

- [ ] **Step 4: Run tests to verify pass**

```bash
.venv/bin/python -m pytest tests/unit/web/sessions/test_persist_compose_turn.py -v -k "happy_path or zero_tool_rows or persists_raw_content or rejects_cross_session_parent_state or accepts_valid_same_session_parent_state or rejects_stale_expected_current_state or accepts_matching_expected_current_state or refuses_async_invocation or async_protocol_dispatch_succeeds_from_async"
.venv/bin/python -m mypy src/elspeth/web/sessions/protocol.py src/elspeth/web/sessions/service.py
```
Expected: PASS for all nine Task-11 tests:
- `test_persist_compose_turn_happy_path`
- `test_persist_compose_turn_zero_tool_rows` (W10a fix — pins the
  assistant-only call shape with empty `redacted_tool_rows` and empty
  `redacted_assistant_tool_calls`)
- `test_persist_compose_turn_persists_raw_content` (B2 fix — the
  primitive accepts the optional `raw_content` argument and persists
  it on the assistant row)
- `test_persist_compose_turn_rejects_cross_session_parent_state`
- `test_persist_compose_turn_accepts_valid_same_session_parent_state`
- `test_persist_compose_turn_rejects_stale_expected_current_state`
- `test_persist_compose_turn_accepts_matching_expected_current_state`
- `test_persist_compose_turn_refuses_async_invocation`
- `test_persist_compose_turn_async_protocol_dispatch_succeeds_from_async`

- [ ] **Step 5: Commit**

```bash
git add src/elspeth/web/sessions/service.py src/elspeth/web/sessions/protocol.py tests/unit/web/sessions/test_persist_compose_turn.py
git commit -m "feat(sessions): add persist_compose_turn happy path (composer-progress-persistence phase 1)"
```

---

## Task 12: `persist_compose_turn` IntegrityError disposition

**Files:**
- Modify: `src/elspeth/web/sessions/service.py`
- Test: `tests/unit/web/sessions/test_persist_compose_turn.py` (extend)

- [ ] **Step 1: Write the failing test**

```python
def test_persist_compose_turn_integrity_error_propagates(service):
    """Duplicate tool_call_id within one session triggers IntegrityError;
    counter increments; helper re-raises (no recovery — spec §4.5)."""
    from sqlalchemy.exc import IntegrityError
    from elspeth.web.sessions._persist_payload import _RedactedToolRow
    from elspeth.web.sessions.telemetry import observed_value

    with service._engine.begin() as conn:
        _make_session(conn, session_id="s7")

    # First turn: creates tool_call_id='dup'
    service.persist_compose_turn(
        session_id="s7",
        assistant_content="",
        redacted_assistant_tool_calls=({"id": "dup", "function": {"name": "x"}},),
        redacted_tool_rows=(_RedactedToolRow("dup", "{}", None),),
        parent_composition_state_id=None,
        expected_current_state_id=None,
        writer_principal="compose_loop",
        plugin_crash_pending=False,
    )

    starting = observed_value(service._telemetry.tool_row_integrity_violation_total)
    with pytest.raises(
        IntegrityError,
        match=(
            r"(UNIQUE.*chat_messages.*session_id.*tool_call_id"
            r"|uq_chat_messages_tool_call_id)"
        ),
    ):
        service.persist_compose_turn(
            session_id="s7",
            assistant_content="",
            redacted_assistant_tool_calls=({"id": "dup", "function": {"name": "x"}},),
            redacted_tool_rows=(_RedactedToolRow("dup", "{}", None),),
            parent_composition_state_id=None,
            expected_current_state_id=None,
            writer_principal="compose_loop",
            plugin_crash_pending=False,
        )
    assert observed_value(service._telemetry.tool_row_integrity_violation_total) == starting + 1


# Spec §4.5 enumerates multiple constraint sources that all flow
# through the same IntegrityError handler in persist_compose_turn.
# The test above covers the partial-unique-tool_call_id source; the
# parametrised test below covers the other reachable source via
# persist_compose_turn's parameter surface.
#
# Sources that are not reachable through public parameters are NOT in
# this matrix because they cannot occur via the entry point:
#
# - role enum violation — persist_compose_turn hardcodes
#   'assistant'/'tool'.
# - **uq_composition_state_version — closed by B1 (Phase 1
#   plan-review synthesis).** ``_StatePayload`` no longer carries a
#   caller-supplied ``version``; ``_insert_composition_state``
#   allocates it under _session_write_lock via
#   ``SELECT COALESCE(MAX(version), 0) + 1 WHERE session_id = :sid``.
#   The constraint is structurally unreachable from
#   ``persist_compose_turn`` — every concurrent allocator serialises
#   on the same lock and observes the previous allocator's
#   committed MAX. The pre-B1 draft of this test asserted the
#   counter SHOULD increment when version=1 was supplied twice;
#   that test codified the fabrication vector B1 closes (every
#   increment was structurally a contention loss masquerading as a
#   Tier-1 audit-integrity violation, with SLO threshold = 0). The
#   replacement test
#   ``test_persist_compose_turn_state_versions_do_not_collide``
#   below pins the post-B1 contract: serial successful persists
#   allocate contiguous versions and the counter MUST remain at its
#   starting value.
#
# ``nonexistent_parent_composition_state`` is deliberately NOT in this
# matrix. Task 11 added the offensive `_assert_state_in_session` guard,
# so a missing parent state is rejected before any INSERT attempts and
# raises RuntimeError with a precise service-boundary diagnostic. Treating
# that as an IntegrityError would make the test pass for the wrong path
# and would double-count a caller contract violation as a Tier-1 DB
# integrity event. The separate RuntimeError regression below pins this.
#
# Closes synthesised review finding M6 / Q-F-04.


@pytest.mark.parametrize(
    "scenario_name, setup_kwargs, expected_match",
    [
        pytest.param(
            "unknown_writer_principal",
            {"writer_principal": "rogue_caller"},
            r"ck_chat_messages_writer_principal",
            id="ck_chat_messages_writer_principal",
        ),
    ],
)
def test_persist_compose_turn_integrity_error_matrix(
    service, scenario_name, setup_kwargs, expected_match,
):
    """Each scenario triggers a distinct §4.5 source via
    persist_compose_turn's parameter surface; all flow through the
    same handler (counter increment + raise). The test asserts both
    the counter increments AND the constraint name appears in the
    raised exception message — without the ``match=``, a wrong
    constraint firing first would false-green this test."""
    from sqlalchemy.exc import IntegrityError
    from elspeth.web.sessions._persist_payload import _RedactedToolRow
    from elspeth.web.sessions.telemetry import observed_value

    with service._engine.begin() as conn:
        _make_session(conn, session_id=f"s_{scenario_name}")

    starting = observed_value(service._telemetry.tool_row_integrity_violation_total)

    base_kwargs = {
        "session_id": f"s_{scenario_name}",
        "assistant_content": "",
        "redacted_assistant_tool_calls": (
            {"id": f"{scenario_name}_tc", "function": {"name": "f"}},
        ),
        "redacted_tool_rows": (
            _RedactedToolRow(f"{scenario_name}_tc", "{}", None),
        ),
        "parent_composition_state_id": None,
        "expected_current_state_id": None,
        "writer_principal": "compose_loop",
        "plugin_crash_pending": False,
    }
    base_kwargs.update(setup_kwargs)

    with pytest.raises(IntegrityError, match=expected_match):
        service.persist_compose_turn(**base_kwargs)

    assert (
        observed_value(service._telemetry.tool_row_integrity_violation_total)
        == starting + 1
    ), f"counter must increment for {scenario_name}"


def test_persist_compose_turn_rejects_missing_parent_state_before_insert(service):
    """A nonexistent parent composition state is a caller contract error,
    not an IntegrityError-source matrix case.

    Task 11's `_assert_state_in_session` guard must reject the missing
    state before the assistant row INSERT. The audit-integrity counter
    must not move because no DB constraint fired and no Tier-1 audit
    corruption was observed.
    """
    from elspeth.web.sessions._persist_payload import _RedactedToolRow
    from elspeth.web.sessions.telemetry import observed_value

    with service._engine.begin() as conn:
        _make_session(conn, session_id="s_missing_parent")

    starting = observed_value(service._telemetry.tool_row_integrity_violation_total)

    with pytest.raises(
        RuntimeError,
        match=(
            r"persist_compose_turn: composition_state_id='doesnotexist' "
            r"does not exist"
        ),
    ):
        service.persist_compose_turn(
            session_id="s_missing_parent",
            assistant_content="",
            redacted_assistant_tool_calls=(
                {"id": "missing_parent_tc", "function": {"name": "f"}},
            ),
            redacted_tool_rows=(
                _RedactedToolRow("missing_parent_tc", "{}", None),
            ),
            parent_composition_state_id="doesnotexist",
            expected_current_state_id=None,
            writer_principal="compose_loop",
            plugin_crash_pending=False,
        )

    assert (
        observed_value(service._telemetry.tool_row_integrity_violation_total)
        == starting
    )


def test_persist_compose_turn_state_versions_do_not_collide(service):
    """B1 (Phase 1 plan-review synthesis): the earlier draft of this
    test was named ``test_persist_compose_turn_duplicate_state_version_propagates``
    and asserted that supplying ``_StatePayload(version=1)`` on two
    successive turns triggered ``uq_composition_state_version`` AND
    incremented ``tool_row_integrity_violation_total``. **That test
    codified the fabrication vector B1 closes** — every IntegrityError
    increment on this constraint is structurally a contention loss
    masquerading as a Tier-1 audit-integrity violation, and the SLO
    threshold for the counter is 0.

    Post-B1 the test is impossible to write: ``_StatePayload`` has no
    ``version`` field, and ``_insert_composition_state`` allocates
    versions under _session_write_lock via
    ``SELECT COALESCE(MAX(version), 0) + 1 WHERE session_id = :sid``.
    Two successive ``persist_compose_turn`` calls for the same session
    each receive a contiguous version (1, then 2); neither raises.
    The counter MUST remain at its starting value.

    This replacement test pins the post-B1 behaviour: serial successful
    persists allocate contiguous versions and never increment the
    integrity counter for the version-collision constraint. The
    concurrent same-state compose contract is exercised on PostgreSQL by
    Task 16's stale-rejection regression."""
    from elspeth.web.sessions._persist_payload import _RedactedToolRow, _StatePayload
    from elspeth.web.sessions.protocol import CompositionStateData
    from elspeth.web.sessions.telemetry import observed_value
    from sqlalchemy import text

    with service._engine.begin() as conn:
        _make_session(conn, session_id="s_ver")

    starting = observed_value(service._telemetry.tool_row_integrity_violation_total)

    service.persist_compose_turn(
        session_id="s_ver",
        assistant_content="",
        redacted_assistant_tool_calls=({"id": "tc_v1", "function": {"name": "f"}},),
        redacted_tool_rows=(
            _RedactedToolRow(
                "tc_v1", "{}",
                _StatePayload(
                    data=CompositionStateData(),
                    derived_from_state_id=None,
                ),
            ),
        ),
        parent_composition_state_id=None,
        expected_current_state_id=None,
        writer_principal="compose_loop",
        plugin_crash_pending=False,
    )

    with service._engine.begin() as conn:
        first_state_id = conn.execute(text(
            "SELECT id FROM composition_states "
            "WHERE session_id='s_ver' ORDER BY version DESC LIMIT 1"
        )).scalar_one()

    # Second turn — pre-B1 this would have collided on
    # uq_composition_state_version because the test supplied
    # version=1 on both turns. Post-B1 the helper allocates
    # version=2 (COALESCE(MAX,0)+1 = 2), so the call succeeds.
    service.persist_compose_turn(
        session_id="s_ver",
        assistant_content="",
        redacted_assistant_tool_calls=({"id": "tc_v2", "function": {"name": "f"}},),
        redacted_tool_rows=(
            _RedactedToolRow(
                "tc_v2", "{}",
                _StatePayload(
                    data=CompositionStateData(),
                    derived_from_state_id=None,
                ),
            ),
        ),
        parent_composition_state_id=None,
        expected_current_state_id=first_state_id,
        writer_principal="compose_loop",
        plugin_crash_pending=False,
    )

    # Counter MUST NOT have moved — there is no version collision
    # because the helper allocated 1 then 2 under the lock.
    assert (
        observed_value(service._telemetry.tool_row_integrity_violation_total)
        == starting
    ), (
        "B1 regression: tool_row_integrity_violation_total incremented "
        "on serial state-version allocation. SLO threshold for this "
        "counter is 0; any increment here is a fabricated Tier-1 alert "
        "and evidence-tampering-class harm under the audit doctrine."
    )

    # And the two states have contiguous versions starting at 1.
    with service._engine.begin() as conn:
        versions = [
            r.version for r in conn.execute(text(
                "SELECT version FROM composition_states "
                "WHERE session_id='s_ver' ORDER BY version"
            ))
        ]
    assert versions == [1, 2], (
        f"B1 regression: per-session version allocation broken; got {versions}"
    )
```

- [ ] **Step 2: Run test to verify it fails**

```bash
.venv/bin/python -m pytest tests/unit/web/sessions/test_persist_compose_turn.py::test_persist_compose_turn_integrity_error_propagates -v
```
Expected: FAIL — counter does not increment because the helper does not catch IntegrityError yet.

- [ ] **Step 3: Wrap the body with IntegrityError catch**

Update `persist_compose_turn` in `src/elspeth/web/sessions/service.py` so the entire `with self._engine.begin() as conn:` body is inside a `try`:

```python
def persist_compose_turn(self, ...) -> _AuditOutcome:
    try:
        with self._engine.begin() as conn:
            # ... existing body ...
    except IntegrityError:
        self._telemetry.tool_row_integrity_violation_total.add(1)
        raise
```

(Add `from sqlalchemy.exc import IntegrityError`.)

- [ ] **Step 4: Run tests to verify pass**

```bash
.venv/bin/python -m pytest tests/unit/web/sessions/test_persist_compose_turn.py -v
```
Expected: PASS for IntegrityError disposition AND happy path still passes.

- [ ] **Step 5: Commit**

```bash
git add src/elspeth/web/sessions/service.py tests/unit/web/sessions/test_persist_compose_turn.py
git commit -m "feat(sessions): persist_compose_turn IntegrityError disposition (composer-progress-persistence phase 1)"
```

---

## Task 13: `persist_compose_turn` OperationalError + audit-failure primacy

**Files:**
- Modify: `src/elspeth/web/sessions/service.py`
- Test: `tests/unit/web/sessions/test_persist_compose_turn.py` (extend) and `tests/unit/web/composer/test_audit_failure_primacy.py` (create)

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/web/composer/test_audit_failure_primacy.py`:

```python
"""Audit-failure primacy disposition (spec §5.2.2 / §5.5 rows 9-10).

Failure injection patches SQLAlchemy's dialect-level ``do_commit``
hook for one COMMIT attempt. This:

1. Complies with spec §8.6 (no mocking of ``persist_compose_turn``'s
   private helpers — ``_acquire_session_advisory_lock``,
   ``_reserve_sequence_range``, ``_insert_chat_message``, and
   ``_insert_composition_state`` exist to be exercised, not mocked).
2. Simulates the **dominant** production trigger named in spec §4.5 —
   COMMIT-time failure (disk full, fsync failure, network partition
   between the last INSERT and COMMIT) — rather than the INSERT-time
   failure the earlier plan draft simulated by mocking
   ``_insert_chat_message``.
3. Exercises the production code's actual ``try: with engine.begin():
   ... except OperationalError: ...`` path end to end. The wrapped
   COMMIT failure surfaces from ``engine.begin().__exit__`` and is
   caught by the outer ``except`` clause in ``persist_compose_turn``.
4. Avoids assigning to ``sqlite3.Connection.commit``. That method is
   read-only on CPython's sqlite3 connection object, so a test that
   patches it fails during setup for the wrong reason.

The earlier draft used ``patch.object(service, "_insert_chat_message",
side_effect=OperationalError(...))``. That violates spec §8.6 (helpers
are mocked) and tests the wrong failure point (INSERT-time).
"""
from __future__ import annotations

import contextlib
import sqlite3
from collections.abc import Iterator

import pytest
from sqlalchemy import Engine
from sqlalchemy.exc import OperationalError

from elspeth.web.sessions.service import SessionServiceImpl
from elspeth.web.sessions._persist_payload import _RedactedToolRow
from elspeth.web.sessions.telemetry import build_sessions_telemetry
import structlog

# Shared ``engine`` fixture and ``_make_session`` helper come from
# ``tests/unit/web/conftest.py`` — the parent-package conftest that
# both the sessions suite and this composer suite share. pytest
# auto-loads it for every test under ``tests/unit/web/...``, so the
# ``engine`` fixture is visible here without further wiring; the
# ``_make_session`` helper is imported explicitly via its absolute
# path (a bare ``from .conftest`` would resolve to
# ``tests/unit/web/composer/conftest.py``, which does not exist —
# synthesised review B5).
from tests.unit.web.conftest import _make_session as _make_session_in_conn


@pytest.fixture
def service(engine, tmp_path):
    return SessionServiceImpl(
        engine, data_dir=tmp_path,
        telemetry=build_sessions_telemetry(),
        log=structlog.get_logger(),
    )


def _make_session(service, session_id):
    """Open a transaction on the service's engine and call the conftest
    helper. Wraps the connection-level helper so audit-primacy tests
    can express setup tersely."""
    with service._engine.begin() as conn:
        _make_session_in_conn(conn, session_id=session_id)


@contextlib.contextmanager
def _force_commit_failure(engine: Engine) -> Iterator[None]:
    """Inject an ``OperationalError`` on the next COMMIT.

    Patches ``engine.dialect.do_commit`` for one call. SQLAlchemy calls
    this hook from ``engine.begin().__exit__``; raising a
    ``sqlite3.OperationalError`` here is wrapped by SQLAlchemy as
    ``sqlalchemy.exc.OperationalError`` and reaches
    ``persist_compose_turn``'s outer OperationalError handler. The
    original hook is restored in ``finally`` so cleanup paths (e.g.
    test teardown) can commit normally.

    SQLite-only — the test suite for audit-failure primacy runs
    against the in-memory SQLite engine. The CL-PP-11 testcontainer
    Postgres test exercises a different scenario (advisory-lock
    contention) and does not require commit-failure injection.
    """
    original_do_commit = engine.dialect.do_commit
    fired = False

    def _fail_once(dbapi_conn: object) -> None:
        nonlocal fired
        if not fired:
            fired = True
            raise sqlite3.OperationalError(
                "simulated COMMIT failure (test injection)"
            )
        original_do_commit(dbapi_conn)

    engine.dialect.do_commit = _fail_once
    try:
        yield
    finally:
        engine.dialect.do_commit = original_do_commit


def test_audit_fail_no_plugin_crash_raises_audit_integrity_error(service):
    """Tool succeeded (plugin_crash_pending=False), audit COMMIT failed:
    ``persist_compose_turn`` must increment the Tier-1 counter AND
    raise :class:`AuditIntegrityError` chained from the original
    ``OperationalError``. Returning a flag would be a doctrine
    violation — the caller could ignore the flag and proceed with
    corrupted audit state. Closes synthesised review finding H1."""
    from elspeth.contracts.errors import AuditIntegrityError, TIER_1_ERRORS
    from elspeth.web.sessions.telemetry import observed_value

    _make_session(service, "p1")
    starting = observed_value(service._telemetry.tool_row_tier1_violation_total)

    with _force_commit_failure(service._engine):
        with pytest.raises(AuditIntegrityError) as exc_info:
            service.persist_compose_turn(
                session_id="p1",
                assistant_content="hi",
                redacted_assistant_tool_calls=(),
                redacted_tool_rows=(),
                parent_composition_state_id=None,
                expected_current_state_id=None,
                writer_principal="compose_loop",
                plugin_crash_pending=False,
            )

    # The original OperationalError is preserved as the chained cause.
    assert exc_info.value.__cause__ is not None
    assert isinstance(exc_info.value.__cause__, OperationalError)

    # Counter increments before the raise — telemetry-after-audit per
    # CLAUDE.md primacy.
    assert (
        observed_value(service._telemetry.tool_row_tier1_violation_total)
        == starting + 1
    )

    # The exception must be in TIER_1_ERRORS so ``except Exception:``
    # blocks cannot silently swallow it.
    assert isinstance(exc_info.value, TIER_1_ERRORS)


def test_audit_fail_during_plugin_crash_records_unwind_failure(service):
    """Tool failed (plugin_crash_pending=True) AND audit COMMIT failed:
    ``persist_compose_turn`` must increment the unwind-audit-failure
    counter and RETURN an outcome with ``unwind_audit_failed=True``.
    The unwind path returns rather than raises because the caller
    already has a captured plugin-crash exception to raise; surfacing
    a separate audit exception here would mask the original tool
    failure. The audit failure is recorded via counter + slog
    (permitted under CLAUDE.md primacy because the audit system
    itself failed)."""
    from elspeth.web.sessions.telemetry import observed_value

    _make_session(service, "p2")
    starting = observed_value(
        service._telemetry.tool_row_persist_failed_during_unwind_total
    )

    with _force_commit_failure(service._engine):
        outcome = service.persist_compose_turn(
            session_id="p2",
            assistant_content="hi",
            redacted_assistant_tool_calls=(),
            redacted_tool_rows=(),
            parent_composition_state_id=None,
            expected_current_state_id=None,
            writer_principal="compose_loop",
            plugin_crash_pending=True,
        )

    assert outcome.assistant_id is None
    assert outcome.unwind_audit_failed is True
    assert (
        observed_value(
            service._telemetry.tool_row_persist_failed_during_unwind_total
        )
        == starting + 1
    )
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv/bin/python -m pytest tests/unit/web/composer/test_audit_failure_primacy.py -v
```
Expected: FAIL.

- [ ] **Step 3: Add OperationalError handler in `persist_compose_turn`**

In `src/elspeth/web/sessions/service.py`, expand the try/except to:

```python
def persist_compose_turn(self, ...) -> _AuditOutcome:
    try:
        with self._engine.begin() as conn:
            # ... existing body ...
    except IntegrityError:
        self._telemetry.tool_row_integrity_violation_total.add(1)
        raise
    except OperationalError as audit_exc:
        if plugin_crash_pending:
            # Tool plugin already crashed; audit unwind also failed.
            # Caller will raise the captured plugin crash — surfacing
            # a separate audit exception here would mask the original
            # tool failure. Record the audit failure via counter +
            # slog (permitted under CLAUDE.md primacy because the
            # audit system itself failed) and return the
            # unwind-failure outcome.
            self._telemetry.tool_row_persist_failed_during_unwind_total.add(1)
            self._log.warning(
                "audit_insert_failed_during_tool_failure_unwind",
                session_id=session_id,
                audit_exc_class=type(audit_exc).__name__,
            )
            return _AuditOutcome(
                assistant_id=None,
                unwind_audit_failed=True,
            )
        # Tier-1 violation: tool succeeded, audit failed. CRASH per
        # CLAUDE.md doctrine. AuditIntegrityError is registered in
        # TIER_1_ERRORS (via @tier_1_error on its declaration in
        # contracts/errors.py) so ``except Exception:`` blocks
        # cannot silently swallow it. The caller has no opportunity
        # to ignore the failure — flag-return would be a doctrine
        # violation per synthesised review finding H1.
        self._telemetry.tool_row_tier1_violation_total.add(1)
        raise AuditIntegrityError(
            f"persist_compose_turn: audit insert failed for "
            f"session_id={session_id!r} with tool succeeded — "
            f"Tier-1 audit corruption (no recovery)"
        ) from audit_exc
```

(Add `from sqlalchemy.exc import OperationalError` and confirm `from elspeth.contracts.errors import AuditIntegrityError` is imported — it already is at the top of `service.py`.)

- [ ] **Step 4: Run tests to verify pass**

```bash
.venv/bin/python -m pytest tests/unit/web/composer/test_audit_failure_primacy.py tests/unit/web/sessions/test_persist_compose_turn.py -v
```
Expected: PASS for primacy tests AND no regression on the previous tests.

- [ ] **Step 5: Commit**

```bash
git add src/elspeth/web/sessions/service.py tests/unit/web/composer/test_audit_failure_primacy.py
git commit -m "feat(sessions): persist_compose_turn OperationalError + audit-failure primacy (composer-progress-persistence phase 1)"
```

---

## Task 14: Atomic `add_message` rewrite — signature, behaviour preservation, protocol, and full call-site sweep

**Why this is ONE atomic task with ONE atomic commit.**

The plan's earlier draft split this work into two tasks (14: change
the service; 15: update routes.py). Six independent reviewers in the
panel review (see plan-revision history) flagged that split as wrong:
between the two commits, every existing positional caller of
`add_message` (routes.py × 6 + tests × current inventory) raises `TypeError` at
call time. CLAUDE.md's no-legacy single-cut policy is the rule that
applies here — breaking changes land in one commit.

The earlier draft also misframed this work as "add a `writer_principal`
kwarg." In reality the rewrite has **six independent dimensions**, and
preserving four pre-rev-4 behaviours requires explicit care. Skipping
any of those preservations would silently regress audit
integrity (worse than a crash, per CLAUDE.md). Both the signature
delta and the behaviour-preservation contract are spelled out below.

---

### 14.1 Signature delta (what changes, what doesn't)

| Dimension | Pre-rev-4 (current) | Rev-4 (this task) | Notes |
|---|---|---|---|
| `session_id` type | `UUID` | `UUID` (unchanged) | Earlier plan draft proposed `str`; that was wrong — protocol.py uses `UUID` for type safety. |
| `role` type | `Literal["user", "assistant", "system", "tool"]` | `Literal["user", "assistant", "system", "tool", "audit"]` (use `ChatMessageRole` alias from `protocol.py`) | `"audit"` is internal-only: audit breadcrumbs that cannot be parented to an assistant. It must be filtered from normal chat responses/history. |
| `content` | `str` | unchanged | |
| `tool_calls` | `Sequence[Mapping[str, Any]] \| None = None` | unchanged | |
| `composition_state_id` | `UUID \| None = None` | unchanged (still `UUID`, not `str`) | |
| `raw_content` | `str \| None = None` | **PRESERVED** (do NOT drop) | Routes 1749 and 2152 pass it; dropping it is silent audit-data loss. |
| `tool_call_id` | (absent) | `str \| None = None` (new) | Required when `role='tool'`, must be `None` otherwise. Audit-only rows use `role='audit'` and carry their envelope in `tool_calls`/`content`, not in `tool_call_id`. |
| `parent_assistant_id` | (absent) | `UUID \| None = None` (new) | Required when `role='tool'`, must be `None` otherwise. Type matches `chat_messages.id` which is a UUID-as-string. Audit-only rows use `role='audit'` when no assistant parent exists. |
| `writer_principal` | (absent) | `str` (new, **REQUIRED keyword-only**) | Must be one of the five enum values in `ck_chat_messages_writer_principal` (`compose_loop`, `route_user_message`, `route_system_message`, `admin_tool`, `session_fork`). |
| Return type | `ChatMessageRecord` | `ChatMessageRecord` (unchanged) | Earlier plan draft proposed `str`; that was wrong — callers consume `.id`, `.created_at`, `.raw_content`. |

**Keyword-only marker placement.** Existing callers use positional
args for `(session_id, role, content)` — that pattern is preserved.
Everything after the first three is keyword-only via a `*,` separator,
so `writer_principal` is required-kwarg, never accidentally swappable
with `tool_calls` or `composition_state_id`.

### 14.2 Behaviour-preservation contract

These four behaviours exist in the current `add_message`
(`service.py:276-325`) and MUST survive the rewrite. Each has an
explicit regression test in this task.

1. **Cross-session guard** — `_assert_state_in_session` (module-level
   function at `service.py:43`) is invoked when `composition_state_id`
   is not `None`, raising `RuntimeError` if the state does not belong
   to the supplied session. This is the offensive-programming pattern
   CLAUDE.md mandates and the only protection against cross-session
   `composition_state_id` misuse before the FK fires.
2. **`updated_at` write-through** — `sessions_table.updated_at` is
   bumped to `now` on every successful insert. UI code at
   `routes.py:189` orders sessions by this column; dropping the write
   silently misorders sessions on the home screen.
3. **`raw_content` persistence** — when `raw_content` is supplied
   (assistant messages whose visible `content` was rewritten by
   runtime preflight redaction), the value is stored verbatim in the
   `raw_content` column. This is the audit-attribution data
   referenced in spec §2 and produced by routes 1749 and 2152.
4. **`ChatMessageRecord` return** — the method returns a populated
   `ChatMessageRecord` (with the freshly-allocated `id`, `created_at`,
   the supplied `raw_content`/`composition_state_id`, and the new
   `tool_call_id` / `parent_assistant_id` linkage fields). Callers at
   routes.py:1393, 1754 (subsequent `assistant_msg.id` reads) and
   tests across `test_fork.py`, `test_routes.py`, `test_service.py`
   consume the returned record.

### 14.3 Files in the atomic commit

This single commit modifies:

- `src/elspeth/web/sessions/protocol.py` — `ChatMessageRole`, `ChatMessageRecord`, and `SessionServiceProtocol.add_message` declaration at lines 258-266 (Stage 0 ground truth).
- `src/elspeth/web/sessions/service.py` — `SessionServiceImpl.add_message` body and `_add_message_sync` helper, **plus `fork_session`'s direct batch insert into `chat_messages_table` at lines ~1115-1209** (see §14.6 below — this is an additional production writer that is invisible to the `\.add_message(` grep below).
- `src/elspeth/web/sessions/routes.py` — 6 production `service.add_message(...)` call sites, including `_persist_tool_invocations` and `_persist_llm_calls`.
- `tests/unit/web/sessions/test_service.py` — ~10 call sites.
- `tests/unit/web/sessions/test_fork.py` — ~38 call sites **plus a new regression test for the `fork_session` batch-insert sweep — see §14.6**.
- `tests/unit/web/sessions/test_routes.py` — ~22 call sites.
- `tests/unit/web/sessions/test_datetime_timezone.py` — 1 call site.
- `tests/unit/web/sessions/test_persist_compose_turn.py` — extends with `add_message` regression tests in this task.

Use `rg -n "\.add_message\(" src tests -g '*.py'`
before the rewrite to confirm the `add_message` call-site count for
your branch. The latest plan-review pass found 86 total matches, with
6 production route calls. The exact number may drift; the grep output,
not the prose count, is the working set.

**ALSO** run `grep -rn "insert(chat_messages_table\|insert(models.chat_messages_table)" src/ --include="*.py"`
to enumerate every direct `chat_messages` writer that bypasses
`add_message`. As of the rev-4 snapshot the only such writer is
`fork_session` — but the grep is the contract. Any new direct-insert
site discovered by this grep MUST be brought into the same atomic
commit as `add_message`'s rewrite, with the same `writer_principal`
and `sequence_no` discipline (see §14.6). Closes B1 from the Phase 1
plan-review synthesis: the `\.add_message(` grep alone is a
necessary-but-insufficient enumeration of `chat_messages` writers.

### 14.4 Production call-site storage mapping

| Site | Role/storage | `writer_principal` | Extra required fields | Rationale |
|---|---|---|---|---|
| `_persist_tool_invocations` (`routes.py:~703`) success-path calls | `"tool"` when caller supplies `parent_assistant_id`; `"audit"` when no assistant parent exists | `"compose_loop"` | tool role: `tool_call_id=invocation.tool_call_id`, `parent_assistant_id=<assistant_msg.id>`; audit role: both NULL | Successful compose/recompose paths have an assistant row and can write real parented tool responses. Failure/convergence/preflight paths do not have an assistant row; those audit breadcrumbs must not violate the parent CHECK, so they use internal `role="audit"`. |
| `_persist_llm_calls` (`routes.py:~752`) | `"audit"` | `"compose_loop"` | `tool_call_id=None`, `parent_assistant_id=None` | LLM-call breadcrumbs are audit-only provider records, not OpenAI tool responses, and have no real tool-call or assistant-parent identity. |
| user message (`routes.py:~1599`) | `"user"` | `"route_user_message"` | no tool linkage | User's incoming message persisted from the route handler. |
| assistant message send path (`routes.py:~1984`) | `"assistant"` | `"compose_loop"` | no tool linkage; preserve `raw_content` | Assistant message produced by the compose loop and persisted from the route handler. Phase 3 will replace this `add_message` call with `persist_compose_turn`. |
| assistant message recompose path (`routes.py:~2431`) | `"assistant"` | `"compose_loop"` | no tool linkage; preserve `raw_content` | Same as send path — second compose-and-persist site. |
| revert system message (`routes.py:~2641`) | `"system"` | `"route_system_message"` | no tool linkage | System message ("Pipeline reverted to version N.") emitted by the state-revert route. |

(Line numbers re-grepped at Phase 1 plan-review stage and may drift.
Always run `rg -n "service\.add_message\(|\.add_message\(" src/elspeth/web/sessions/routes.py`
before editing rather than trusting any number written here.)

The assistant send/recompose sites pass `raw_content=result.raw_assistant_content`;
the rewrite preserves this — `add_message` continues to accept and
persist `raw_content`. (Earlier plan draft would have silently
dropped it; see §14.1.)

**Audit-role filtering requirement.** Update
`_is_composer_audit_tool_message` / `_composer_conversation_messages`
so `role="audit"` rows are excluded from normal chat responses and
from composer prompt history. Add a route/unit test proving an audit
row with `_kind="llm_call_audit"` is stored but absent from
`GET /api/sessions/{id}/messages` and `_composer_chat_history(...)`.
The test must create or load an actual `chat_messages.role='audit'`
row through the service/database path before calling the route helper;
do not satisfy this requirement by testing only a synthetic in-memory
`ChatMessageRecord` list.

The filtering predicate must treat `role="audit"` as internal even when
its envelope looks like a tool/audit breadcrumb:

```python
if message.role == "audit":
    return True
if message.role != "tool":
    return False
```

Update both the public route response path and the prompt-history helper;
fixing only one leaves either API leakage or hidden prompt pollution.

### 14.5 Test-suite call-site migration

The test suite has ~71 `service.add_message(...)` calls. Most are
positional `service.add_message(session.id, "user", "msg")` patterns.
The mechanical migration: append `, writer_principal=<value>` to
every call, choosing the value by role:

| Test message role | `writer_principal` |
|---|---|
| `"user"` | `"route_user_message"` |
| `"assistant"` | `"compose_loop"` |
| `"system"` | `"route_system_message"` |
| `"tool"` | `"compose_loop"` (tool rows are produced by the compose loop) |
| `"audit"` | `"compose_loop"` (audit-only composer breadcrumbs) |

Tests that exercise auth-scoped or admin-tool behaviour may need a
different value; treat the table above as the default and override
explicitly where the test's intent calls for it.

The mechanical migration is deterministic enough that a one-shot sed
or codemod can do most of it, but a hand-review is required for tests
that *intentionally* exercise a wrong-principal scenario (none exist
today; if any test asserts a specific principal value in the audit
trail, it owns the explicit choice).

- [ ] **Step 1: Confirm the call-site inventory before editing**

```bash
rg -n "\.add_message\(" src/elspeth/web/sessions/routes.py src/elspeth/web/sessions/protocol.py src/elspeth/web/sessions/service.py tests/unit/web/sessions/ tests/unit/web/blobs/ tests/integration/web/ -g '*.py'
```
Expected on the reviewed snapshot: 86 total matches, including 6
production route helper/call sites, 1 protocol declaration, 1 service
definition, and existing tests. The grep output is the implementer's
working set for this commit; paste the count into the PR body.

- [ ] **Step 2: Write the regression tests pinning the four preserved behaviours**

Add to `tests/unit/web/sessions/test_persist_compose_turn.py` (the
file already imports `_make_session` from conftest):

```python
# All tests below are async because ``add_message`` and
# ``save_composition_state`` are coroutines. The project ships
# ``pytest-asyncio>=1.0,<2`` (declared in pyproject.toml dev deps;
# the ``asyncio`` marker is registered) so ``@pytest.mark.asyncio
# async def`` is the idiomatic shape. Calling ``asyncio.run`` from a
# sync test function fights pytest-asyncio's event-loop management
# and fails outright when ``asyncio_mode = "auto"`` is configured —
# the synthesised review's M3 finding.


@pytest.mark.asyncio
async def test_add_message_preserves_assert_state_in_session_guard(service):
    """Rev-4 must NOT silently drop the cross-session composition_state_id guard
    that pre-rev-4 add_message enforced via _assert_state_in_session.
    Closes synthesised review finding SA-9."""
    from uuid import uuid4
    from elspeth.web.sessions.protocol import CompositionStateData

    sid_a = uuid4()
    sid_b = uuid4()
    with service._engine.begin() as conn:
        _make_session(conn, session_id=str(sid_a))
        _make_session(conn, session_id=str(sid_b))

    # Save a state on session A.
    state_a = await service.save_composition_state(sid_a, CompositionStateData())

    # Attempt to add a message on session B referencing A's state.
    # Pre-rev-4 raised RuntimeError via _assert_state_in_session.
    # Rev-4 MUST raise the same error from the same guard.
    with pytest.raises(RuntimeError, match="cross-session reference"):
        await service.add_message(
            sid_b,
            "user",
            "should fail",
            composition_state_id=state_a.id,
            writer_principal="route_user_message",
        )


@pytest.mark.asyncio
async def test_add_message_preserves_updated_at_write(service):
    """Rev-4 must continue to bump sessions.updated_at on every insert.
    UI code orders sessions by updated_at; a missing write silently
    misorders the home screen."""
    import asyncio
    from uuid import uuid4
    from sqlalchemy import select
    from elspeth.web.sessions import models

    sid = uuid4()
    with service._engine.begin() as conn:
        _make_session(conn, session_id=str(sid))
        before = conn.execute(
            select(models.sessions_table.c.updated_at).where(
                models.sessions_table.c.id == str(sid)
            )
        ).scalar_one()

    # Yield to the event loop and a wall-clock millisecond so
    # ``updated_at`` can advance observably. ``asyncio.sleep`` is
    # the async-test idiom; ``time.sleep`` would block the loop.
    await asyncio.sleep(0.001)

    await service.add_message(
        sid, "user", "hi", writer_principal="route_user_message"
    )

    with service._engine.begin() as conn:
        after = conn.execute(
            select(models.sessions_table.c.updated_at).where(
                models.sessions_table.c.id == str(sid)
            )
        ).scalar_one()
    assert after > before, "sessions.updated_at must advance after add_message"


@pytest.mark.asyncio
async def test_add_message_preserves_raw_content(service):
    """Rev-4 must persist raw_content when supplied. Routes 1749 and
    2152 pass raw_content=result.raw_assistant_content for assistant
    messages whose visible content was rewritten by preflight
    redaction; dropping it would silently lose audit attribution."""
    from uuid import uuid4

    sid = uuid4()
    with service._engine.begin() as conn:
        _make_session(conn, session_id=str(sid))

    record = await service.add_message(
        sid, "assistant", "redacted",
        raw_content="original LLM output",
        writer_principal="compose_loop",
    )
    assert record.raw_content == "original LLM output"

    # Double-check via DB query — record state must reflect persisted state.
    from sqlalchemy import select
    from elspeth.web.sessions import models
    with service._engine.begin() as conn:
        row = conn.execute(
            select(models.chat_messages_table.c.raw_content).where(
                models.chat_messages_table.c.id == str(record.id)
            )
        ).scalar_one()
    assert row == "original LLM output"


@pytest.mark.asyncio
async def test_add_message_returns_chat_message_record(service):
    """Return type MUST stay ChatMessageRecord. Callers consume
    .id, .created_at, .raw_content — a str return breaks every one."""
    from uuid import uuid4
    from elspeth.web.sessions.protocol import ChatMessageRecord

    sid = uuid4()
    with service._engine.begin() as conn:
        _make_session(conn, session_id=str(sid))

    result = await service.add_message(
        sid, "user", "hi", writer_principal="route_user_message"
    )
    assert isinstance(result, ChatMessageRecord)
    assert result.session_id == sid
    assert result.role == "user"
    assert result.content == "hi"
    assert result.created_at is not None


@pytest.mark.asyncio
async def test_add_message_requires_writer_principal(service):
    """Rev-4 breaking change: writer_principal is required keyword-only."""
    from uuid import uuid4

    sid = uuid4()
    with service._engine.begin() as conn:
        _make_session(conn, session_id=str(sid))
    with pytest.raises(TypeError, match="writer_principal"):
        await service.add_message(sid, "user", "hi")  # type: ignore[call-arg]


@pytest.mark.asyncio
async def test_add_message_rejects_unknown_writer_principal(service):
    """The CHECK constraint backs the type system: an unknown
    writer_principal value MUST raise IntegrityError at write time
    (the schema is the load-bearing enforcement; the Python type is
    `str` for forward-compatibility with future enum extensions)."""
    from uuid import uuid4
    from sqlalchemy.exc import IntegrityError

    sid = uuid4()
    with service._engine.begin() as conn:
        _make_session(conn, session_id=str(sid))
    with pytest.raises(IntegrityError, match="ck_chat_messages_writer_principal"):
        await service.add_message(
            sid, "user", "hi", writer_principal="rogue_writer"
        )
```

- [ ] **Step 3: Run the regression tests to verify they fail**

```bash
.venv/bin/python -m pytest tests/unit/web/sessions/test_persist_compose_turn.py -v -k add_message
```
Expected: every regression test FAILS, with errors mentioning either
the missing `writer_principal` argument or the unchanged
`add_message` signature.

- [ ] **Step 4: Update `SessionServiceProtocol.add_message`**

In `src/elspeth/web/sessions/protocol.py`, first extend the role alias
so the internal audit breadcrumb role is typed:

```python
ChatMessageRole = Literal["user", "assistant", "system", "tool", "audit"]
```

Append the new linkage fields to `ChatMessageRecord` so every read path
hydrates the columns Task 1 added:

```python
    tool_call_id: str | None = None
    parent_assistant_id: UUID | None = None
```

`tool_call_id` is the provider/tool-call identifier and remains a
string. `parent_assistant_id` is a chat-message primary key exposed by
the service layer the same way `id`, `session_id`, and
`composition_state_id` are exposed: as `UUID` values, not raw DB strings.

Then replace the existing declaration at lines 258-266 with:

```python
async def add_message(
    self,
    session_id: UUID,
    role: ChatMessageRole,
    content: str,
    *,
    writer_principal: str,
    tool_calls: Sequence[Mapping[str, Any]] | None = None,
    composition_state_id: UUID | None = None,
    raw_content: str | None = None,
    tool_call_id: str | None = None,
    parent_assistant_id: UUID | None = None,
) -> ChatMessageRecord: ...
```

- [ ] **Step 5: Rewrite `SessionServiceImpl.add_message`**

In `src/elspeth/web/sessions/service.py`, replace the existing
`add_message` (lines 276-325) with:

```python
async def add_message(
    self,
    session_id: UUID,
    role: ChatMessageRole,
    content: str,
    *,
    writer_principal: str,
    tool_calls: Sequence[Mapping[str, Any]] | None = None,
    composition_state_id: UUID | None = None,
    raw_content: str | None = None,
    tool_call_id: str | None = None,
    parent_assistant_id: UUID | None = None,
) -> ChatMessageRecord:
    """Add a chat message and update the session's updated_at.

    BREAKING CHANGE in rev 4: ``writer_principal`` is now a required
    keyword-only argument (must be one of the values listed in the
    ``ck_chat_messages_writer_principal`` CHECK constraint). All
    callers were updated atomically with this signature change.

    Preserved behaviours from pre-rev-4:
    - ``_assert_state_in_session`` cross-session guard fires when
      ``composition_state_id`` is not None.
    - ``sessions_table.updated_at`` is bumped to ``now``.
    - ``raw_content`` is persisted verbatim when supplied.
    - Returns ``ChatMessageRecord``, not just an id string.

    New in rev 4:
    - ``sequence_no`` is allocated under ``_session_write_lock``
      (PostgreSQL advisory lock or SQLite per-session process lock),
      replacing the implicit "last write wins" ordering.
    - ``tool_call_id`` and ``parent_assistant_id`` MUST be set when
      ``role='tool'`` and MUST be None otherwise; the
      ``ck_chat_messages_tool_call_id_role`` and
      ``ck_chat_messages_parent_role`` CHECK constraints enforce
      this at write time.
    """
    now = self._now()
    sid = str(session_id)
    csid = str(composition_state_id) if composition_state_id else None
    pid = str(parent_assistant_id) if parent_assistant_id else None
    msg_id_holder: dict[str, str] = {}

    def _sync() -> None:
        with self._engine.begin() as conn:
            # Cross-session guard preserved from pre-rev-4. Module-level
            # function at service.py:43, NOT a method on self.
            if csid is not None:
                _assert_state_in_session(
                    conn,
                    state_id=csid,
                    expected_session_id=sid,
                    caller="add_message",
                )
            # Rev-4 sequence allocation under the session write lock.
            with self._session_write_lock(conn, sid):
                seq = self._reserve_sequence_range(conn, sid, count=1)
                msg_id_holder["id"] = self._insert_chat_message(
                    conn,
                    session_id=sid,
                    role=role,
                    content=content,
                    raw_content=raw_content,
                    # Same JSON-serialisation rationale as the
                    # persist_compose_turn site — see L18 comment there.
                    tool_calls=deep_thaw(tool_calls) if tool_calls else None,
                    sequence_no=seq,
                    writer_principal=writer_principal,
                    composition_state_id=csid,
                    tool_call_id=tool_call_id,
                    parent_assistant_id=pid,
                    created_at=now,
                )
            # updated_at preserved from pre-rev-4.
            conn.execute(
                update(sessions_table)
                .where(sessions_table.c.id == sid)
                .values(updated_at=now)
            )

    await self._run_sync(_sync)

    return ChatMessageRecord(
        id=UUID(msg_id_holder["id"]),
        session_id=session_id,
        role=role,
        content=content,
        raw_content=raw_content,
        tool_calls=tool_calls,
        created_at=now,
        composition_state_id=composition_state_id,
        tool_call_id=tool_call_id,
        parent_assistant_id=parent_assistant_id,
    )
```

- [ ] **Step 6: Update routes.py call sites and audit helpers**

Apply all six updates from §14.4. The two helper functions are the
non-mechanical part and must be changed first:

```python
async def _persist_tool_invocations(
    service: SessionServiceProtocol,
    session_id: UUID,
    tool_invocations: tuple[ComposerToolInvocation, ...],
    composition_state_id: UUID | None,
    *,
    parent_assistant_id: UUID | None = None,
) -> None:
    """Persist tool/audit breadcrumbs without violating role=tool parent invariants."""
    for invocation in tool_invocations:
        # ... existing content selection ...
        role: ChatMessageRole = "tool" if parent_assistant_id is not None else "audit"
        await service.add_message(
            session_id,
            role,
            content,
            tool_calls=[audit_envelope(invocation)],
            composition_state_id=composition_state_id,
            writer_principal="compose_loop",
            tool_call_id=invocation.tool_call_id if role == "tool" else None,
            parent_assistant_id=parent_assistant_id,
        )


async def _persist_llm_calls(
    service: SessionServiceProtocol,
    session_id: UUID,
    llm_calls: tuple[ComposerLLMCall, ...],
    composition_state_id: UUID | None,
) -> None:
    """Persist provider-call audit breadcrumbs as internal audit rows."""
    for call in llm_calls:
        # ... existing content selection ...
        await service.add_message(
            session_id,
            "audit",
            content,
            tool_calls=[llm_call_audit_envelope(call)],
            composition_state_id=composition_state_id,
            writer_principal="compose_loop",
        )
```

Then update success-path calls after assistant persistence to pass
`parent_assistant_id=assistant_msg.id`:

```bash
# send_message / recompose success paths:
#   await _persist_tool_invocations(
#       service,
#       session.id,
#       result.tool_invocations,
#       post_compose_state_id,
#  +    parent_assistant_id=assistant_msg.id,
#   )

# User-message persist
#   user_msg = await service.add_message(
#       session.id, "user", body.content,
#       composition_state_id=pre_send_state_id,
#  +    writer_principal="route_user_message",
#   )

# Assistant messages after compose/recompose
#   assistant_msg = await service.add_message(
#       session.id, "assistant", result.message,
#       composition_state_id=post_compose_state_id,
#       raw_content=result.raw_assistant_content,
#  +    writer_principal="compose_loop",
#   )

# System message after state revert
#   await service.add_message(
#       session.id,
#       role="system",
#       content=f"Pipeline reverted to version {original_state.version}.",
#  +    writer_principal="route_system_message",
#   )
```

(Use the actual current line numbers from your branch — the four
sites above match the snapshot taken in Stage 0 of the plan revision.)

- [ ] **Step 7: Sweep test-suite call sites**

Apply the table from §14.5 to every test in:
- `tests/unit/web/sessions/test_service.py` (~10 sites)
- `tests/unit/web/sessions/test_fork.py` (~38 sites)
- `tests/unit/web/sessions/test_routes.py` (~22 sites)
- `tests/unit/web/sessions/test_datetime_timezone.py` (1 site)

For each call, append `, writer_principal=<value>` per the role-keyed
default table. Hand-review any test whose intent is to exercise a
specific principal (none exist today; new ones may have been added
after this plan was authored).

- [ ] **Step 8: Run the full sessions unit and integration suites**

```bash
.venv/bin/python -m pytest tests/unit/web/sessions/ tests/unit/web/blobs/test_routes.py tests/integration/web/test_inv_audit_ahead_backward.py -v
.venv/bin/python -m mypy src/elspeth/web/sessions/service.py src/elspeth/web/sessions/protocol.py src/elspeth/web/sessions/routes.py
```
Expected: every test PASS; mypy clean.

- [ ] **Step 9: Single atomic commit**

```bash
git add src/elspeth/web/sessions/service.py \
        src/elspeth/web/sessions/protocol.py \
        src/elspeth/web/sessions/routes.py \
        tests/unit/web/sessions/test_persist_compose_turn.py \
        tests/unit/web/sessions/test_service.py \
        tests/unit/web/sessions/test_fork.py \
        tests/unit/web/sessions/test_routes.py \
        tests/unit/web/sessions/test_datetime_timezone.py
git commit -m "feat(sessions)!: rev-4 add_message rewrite — required writer_principal, sequence_no allocation, full call-site sweep (composer-progress-persistence phase 1)

BREAKING: SessionServiceProtocol.add_message and SessionServiceImpl.add_message
now require a keyword-only \`writer_principal\` argument matching the
\`ck_chat_messages_writer_principal\` CHECK enum. Adds optional
\`tool_call_id\` and \`parent_assistant_id\` arguments (required when
role='tool', forbidden otherwise — enforced by CHECK constraints).
Adds \`sequence_no\` allocation under _session_write_lock.

PRESERVED behaviours: _assert_state_in_session cross-session guard,
sessions.updated_at write-through, raw_content persistence,
ChatMessageRecord return type. Each preserved behaviour has an
explicit regression test.

All callers (6 production route/helper sites in routes.py + 1
production batch-write site in fork_session [§14.6] + the current test
inventory + the SessionServiceProtocol declaration) updated in this
single atomic commit per the no-legacy single-cut policy.

Also switches \`get_messages\` ORDER BY from \`created_at\` to
\`sequence_no\` and removes the \`fork_session\` microsecond-offset
workaround that papered over fast-SQLite ordering ambiguity (§14.7).
The read-side and write-side switch in the same atomic commit.
"
```

---

### 14.6 `fork_session` batch-insert sweep (closes B1)

**Why this section exists.** The Phase 1 plan-review synthesis (B1)
flagged that `fork_session` at `service.py:1115-1209` writes
`chat_messages` rows via a direct `conn.execute(insert(chat_messages_table), msg_records_data)`
batch, completely bypassing `add_message`. The plan's route-level
inventory already covers six `service.add_message(...)` writers; this
direct insert is the additional production `chat_messages` writer that
the `.add_message(` grep cannot see. After Task 1 lands NOT NULL `writer_principal`
and `sequence_no`, every `/fork_from_message` request would crash at
runtime with `IntegrityError`. This sweep brings `fork_session` into
the same atomic commit as the `add_message` rewrite.

**File:** `src/elspeth/web/sessions/service.py` (`fork_session`, lines ~1115-1209).

**Change 1 — populate `writer_principal` for every row in `msg_records_data`.**

The fork transaction builds three categories of rows:

| Row category | Source line range | `writer_principal` value | Rationale |
|---|---|---|---|
| Copied source-session messages | ~1115-1129 | role-keyed per §14.5 table (`"user"`→`"route_user_message"`, `"assistant"`→`"compose_loop"`, `"system"`→`"route_system_message"`, `"tool"`→`"compose_loop"`, `"audit"`→`"compose_loop"`) | These rows came from the source session's history; the role-keyed default is the closest available approximation. **Fidelity follow-up (OQ):** preserving the *original* source row's `writer_principal` would be more truthful, but `ChatMessageRecord` does not currently expose that field. File an OQ ticket to extend `ChatMessageRecord` and switch to source-preservation in a follow-up phase. |
| New synthetic system message ("Conversation forked from an earlier point.") | ~1131-1143 | `"session_fork"` (new enum value added to `ck_chat_messages_writer_principal` CHECK in Task 1) | This row is unambiguously authored by the fork operation, not by the route handler that originally accepted user input. |
| New edited user message | ~1149-1161 | `"session_fork"` | Same — the fork operation, not the route handler, is the authoritative writer. The `composition_state_id` correctly points at the COPIED state in the new session (current behaviour preserved). |

**Change 2 — allocate `sequence_no` for every row under the new session write lock.**

Inside the `_sync()` closure (currently lines ~1169-1211), insert the
following sequence between the session insert (step 1, ~1173-1184)
and the composition-state insert (step 2, ~1186-1205):

```python
# Step 1.5 — enter the session write-lock context for the NEW session
# and reserve a contiguous sequence range covering every row in
# msg_records_data. The lock is technically uncontended because
# new_session_id_str was minted seconds ago and no other writer can
# know it yet, but the helpers' precondition contract requires the
# lock context; honouring the contract uniformly is cheaper than
# reasoning about exceptions. Keep the later composition-state helper
# call in this same context so state version and chat sequence
# allocation stay under one SQLite/PostgreSQL serialization boundary.
with self._session_write_lock(conn, new_session_id_str):
    base_seq = self._reserve_sequence_range(
        conn, new_session_id_str, count=len(msg_records_data)
    )
    for offset, record in enumerate(msg_records_data):
        record["sequence_no"] = base_seq + offset
    # Insert the copied composition state and chat rows while still
    # inside this context.
```

The order of `msg_records_data` already encodes the intended chat
sequence: copied messages first (in source order), then the system
fork notice, then the new user message. Assigning `base_seq + offset`
in list order locks that ordering into the canonical
`sequence_no` key — which is what `get_messages` will order by after
B2 lands.

**Change 3 — preserve `tool_call_id` and rewrite `parent_assistant_id` for copied tool rows.**

Tool rows in the source session have non-NULL `tool_call_id` and
`parent_assistant_id`. The current copy loop omits both. After Task 1
lands the CHECK constraints `(role = 'tool') = (tool_call_id IS NOT NULL)`
and `(role = 'tool') = (parent_assistant_id IS NOT NULL)`, omitting
either field on a tool row crashes with `IntegrityError`. Copying the
source `parent_assistant_id` verbatim is also wrong: the copied tool row
lives in a new session, so its parent must be the copied assistant row's
new id. Build a source-id to copied-id map for in-slice assistant rows
before inserting tool rows:

```python
source_to_copied_assistant_id: dict[str, str] = {}
for msg in messages_to_copy:
    copied_msg_id = str(uuid.uuid4())
    if msg.role == "assistant":
        source_to_copied_assistant_id[str(msg.id)] = copied_msg_id

for msg in messages_to_copy:
    copied_msg_id = (
        source_to_copied_assistant_id[str(msg.id)]
        if msg.role == "assistant"
        else str(uuid.uuid4())
    )
    copied_parent_assistant_id: str | None = None
    if msg.role == "tool":
        if msg.parent_assistant_id is None:
            raise RuntimeError(
                f"fork_session: tool message id={msg.id} has no parent assistant"
            )
        copied_parent_assistant_id = source_to_copied_assistant_id.get(
            str(msg.parent_assistant_id)
        )
        if copied_parent_assistant_id is None:
            raise RuntimeError(
                "fork slice excludes parent assistant "
                f"of tool message id={msg.id}"
            )

    msg_records_data.append(
        {
            "id": copied_msg_id,
            "session_id": new_session_id_str,
            "role": msg.role,
            "content": msg.content,
            "raw_content": msg.raw_content,
            "tool_calls": deep_thaw(msg.tool_calls) if msg.tool_calls else None,
            "tool_call_id": msg.tool_call_id,            # NEW
            "parent_assistant_id": copied_parent_assistant_id,  # NEW, rewritten
            "writer_principal": _ROLE_TO_WRITER_PRINCIPAL[msg.role],  # NEW
            "created_at": msg.created_at,
            "composition_state_id": None,  # Don't reference source session states
        }
    )
```

This requires `ChatMessageRecord` to expose `tool_call_id` and
`parent_assistant_id` (both are post-Task-1 columns). Confirm at the
top of T14 that the protocol dataclass has been extended to include
them — if not, this is the commit that extends it (it's part of the
same atomic cut).

The exact field declarations to append to
`src/elspeth/web/sessions/protocol.py`'s `ChatMessageRecord` dataclass
(end-of-class style — match the existing field placement for
`raw_content`, `tool_calls`, `composition_state_id`):

```python
    tool_call_id: str | None = None
    parent_assistant_id: UUID | None = None
```

Type rationale: `tool_call_id` is a provider/tool-call identifier and
stays `str | None`. `parent_assistant_id` references
`chat_messages.id`, which the service layer already exposes as `UUID`
on `ChatMessageRecord.id`; hydrate it to `UUID | None` in every
constructor (`add_message`, `get_messages`, and `fork_session`
returns). **No `freeze_fields` call is required for these two fields**
— both are scalar/immutable values, so the `frozen=True` slot itself is
sufficient deep-immutability guard (per CLAUDE.md's "Scalar-Only Fields
Need No Guard"). The existing `__post_init__` only calls
`freeze_fields(self, "tool_calls")` because `tool_calls` is the only
container field. Do **not** add the new fields to that guard list.

**`parent_assistant_id` slice-boundary caveat.** A copied tool row
whose `parent_assistant_id` references an assistant message OUTSIDE
the `[:fork_idx]` slice will FK-fail when the batch insert tries to
resolve the reference inside the new session. The current `fork_session`
slice logic does not detect this; after Task 1 lands the column, the
failure mode is a deferred `IntegrityError` at commit time. **In
scope for this task:** detect the condition before the batch insert
and raise a precise `RuntimeError("fork slice excludes parent assistant
of tool message id=...")` with the offending message ID. The
offensive-programming policy requires a named pre-check rather than
a generic FK failure (parallel to B5's `_assert_state_in_session`
guard).

**Change 4 — the new fork-time `created_at` values.**

The current code stamps `now` for the system fork notice (line ~1140)
and `now + timedelta(microseconds=1)` for the user edit (line ~1158)
to bypass the same-microsecond ordering ambiguity. After `sequence_no`
becomes the canonical ordering key, the microsecond offset is no
longer needed. Build the fork rows with the writer/linkage/sequence
fields described here; §14.7 removes the offset in the same atomic T14
commit so the write-side and read-side ordering changes land together.

**Helper constant.** Add to `service.py` near the top of the module
(after the imports). **`MappingProxyType` is not currently imported by
`service.py`**, so this commit must also add `from types import
MappingProxyType` to the module imports (alphabetical with the existing
`from datetime import UTC, datetime, timedelta` / `from pathlib import
Path` block) — the constant declaration below relies on it.

```python
_ROLE_TO_WRITER_PRINCIPAL: Mapping[str, str] = MappingProxyType({
    "user": "route_user_message",
    "assistant": "compose_loop",
    "system": "route_system_message",
    "tool": "compose_loop",
    "audit": "compose_loop",
})
```

The `MappingProxyType` wrap matches the project's `freeze_fields`
discipline — module-level mappings that callers might mutate are a
defect class CLAUDE.md's frozen-dataclass section calls out.

**Required regression tests** (add to `tests/unit/web/sessions/test_fork.py`):

1. `test_fork_session_assigns_writer_principal_per_role` — fork a
   session that contains user, assistant, system, and tool rows;
   assert each copied row's `writer_principal` matches the role-keyed
   default, the new system fork notice has `writer_principal="session_fork"`,
   and the new edited user message has `writer_principal="session_fork"`.
2. `test_fork_session_assigns_contiguous_sequence_no` — fork a session
   with N copied messages; assert the resulting `chat_messages.sequence_no`
   values for the new session are exactly `[1, 2, ..., N+2]` (N copied
   plus 2 fork-time inserts), with no gaps and no out-of-order values.
3. `test_fork_session_preserves_tool_call_id_and_parent` — fork a session
   that contains a tool row referencing an in-slice assistant; assert
   the copied tool row's `tool_call_id` and `parent_assistant_id` are
   carried over (with `parent_assistant_id` rewritten to point at the
   COPIED assistant message's new ID, since the source ID is no longer
   valid in the new session).
4. `test_fork_session_rejects_tool_with_out_of_slice_parent` — fork a
   session where the slice `[:fork_idx]` excludes the assistant message
   that an in-slice tool row depends on; assert `RuntimeError` with the
   precise "fork slice excludes parent assistant" message.
5. `test_fork_session_rejects_with_admin_tool_writer_principal_when_not_admin`
   — sanity check that the role-keyed default does not silently
   downgrade an `admin_tool` source row; document expected behaviour
   (currently `admin_tool` is reserved for future use and produces no
   rows, so this test is a no-op pinning that future behaviour requires
   a deliberate decision).

The first four tests are required Green-bar additions. Test 5 is a
documentation pin — annotate it with `pytest.skip(reason="admin_tool
reserved")` if the source path produces no `admin_tool` rows today.

---

### 14.7 `get_messages` ordering switch + `fork_session` microsecond-hack removal (closes B2)

**Why this section exists.** The Phase 1 plan-review synthesis (B2)
flagged that `get_messages` (`service.py:340`) currently orders by
`chat_messages.created_at`, while `persist_compose_turn` stamps every
row in a single turn (assistant + N tool rows) with one shared `now`
value. On fast SQLite paths the rows share a microsecond and
`get_messages` returns them in arbitrary order. Once `sequence_no`
exists (Task 1) and is allocated under the session-scoped advisory
lock (Tasks 7-9), it is the canonical monotonic ordering key — and
`created_at` becomes informational only.

This sub-section folds the ordering switch into the same atomic T14
commit that introduces `sequence_no` allocation, so the read-side and
write-side switch at the same boundary. Splitting them across commits
would leave a window in which the new column exists but reads are
still ordered by the old key.

**Change 1 — `get_messages` ORDER BY clause (`service.py:340`).**

```python
# BEFORE (service.py:336-341, current behaviour)
result = conn.execute(
    select(chat_messages_table)
    .where(chat_messages_table.c.session_id == str(session_id))
    .order_by(chat_messages_table.c.created_at)
)

# AFTER
result = conn.execute(
    select(chat_messages_table)
    .where(chat_messages_table.c.session_id == str(session_id))
    .order_by(chat_messages_table.c.sequence_no)
)
```

`sequence_no` is `NOT NULL` (Task 1), so the ORDER BY is total. The
allocator (`_reserve_sequence_range` under _session_write_lock) is the
single arbiter of ordering, and the per-session unique index
`ix_chat_messages_session_sequence` (Task 1) makes the ordering key
both unique and dense within a session.

In the same `get_messages` edit, hydrate the new linkage fields on every
returned `ChatMessageRecord`:

```python
ChatMessageRecord(
    # ... existing fields ...
    tool_call_id=row.tool_call_id,
    parent_assistant_id=UUID(row.parent_assistant_id)
    if row.parent_assistant_id is not None
    else None,
)
```

This is not optional: §14.6's `fork_session` copy loop reads
`msg.tool_call_id` and `msg.parent_assistant_id`, and Phase 3/4 recovery
surfaces rely on the service API carrying the same parentage the DB
enforces.

**Change 2 — Remove the `fork_session` microsecond-offset workaround.**

Once `sequence_no` is the read-side ordering key, the microsecond
offset at `service.py:~1149-1161` (the new edited user message's
`created_at = now + timedelta(microseconds=1)` and the explanatory
comment block at lines ~1144-1148) is no longer needed. The list
order of `msg_records_data` already encodes the intended chat
sequence; §14.6's `base_seq + offset` assignment locks that order
into `sequence_no`.

```python
# BEFORE (service.py:~1144-1161)
# New edited user message — provenance points to COPIED state, not source.
# Offset by 1 microsecond so get_messages() ordering is deterministic
# (system note before user turn).  Without this, SQLite/Postgres can
# return the two rows in either order since they share created_at.
# raw_content is None: this is a new user-authored message, not an LLM turn.
new_user_msg_id = str(uuid.uuid4())
msg_records_data.append(
    {
        "id": new_user_msg_id,
        "session_id": new_session_id_str,
        "role": "user",
        "content": new_message_content,
        "raw_content": None,
        "tool_calls": None,
        "created_at": now + timedelta(microseconds=1),
        "composition_state_id": copied_state_id_str,
    }
)

# AFTER
# New edited user message — provenance points to COPIED state, not source.
# created_at = now is correct here: ordering is enforced by sequence_no
# (allocated in §14.6's base_seq + offset loop), not by created_at.
# raw_content is None: this is a new user-authored message, not an LLM turn.
new_user_msg_id = str(uuid.uuid4())
msg_records_data.append(
    {
        "id": new_user_msg_id,
        "session_id": new_session_id_str,
        "role": "user",
        "content": new_message_content,
        "raw_content": None,
        "tool_calls": None,
        "created_at": now,
        "composition_state_id": copied_state_id_str,
    }
)
```

If `fork_session`'s import block currently includes `from datetime
import timedelta` solely for this microsecond offset, remove the
import in the same edit. (Other usages — `grep -n "timedelta" service.py`
— may keep it alive; verify before deleting.)

**Change 3 — read-path consumers that depend on chat-message ordering.**

`grep -rn "get_messages\|chat_messages_table.*order_by" src/ tests/ --include="*.py"`
to enumerate every consumer of `get_messages`. As of the rev-4
snapshot:

- `src/elspeth/web/sessions/routes.py:412` (`_composer_chat_history`)
  reads `get_messages` output to build the LLM input. Behaviour is
  unchanged — the order it already expects (assistant before its tool
  rows, system before user, copied messages before fork-time inserts)
  is exactly what `sequence_no` enforces.
- Phase 4 recovery panel (not yet implemented) will consume
  `get_messages` and depends on stable intra-turn ordering. The
  ordering switch is its prerequisite.

No call site is broken by the change; the switch is a strict
strengthening of the previously implicit (and on fast SQLite,
unreliable) `created_at` ordering.

**Required regression test (added inside T15's Step 1 — see T15
amendment).** A new integration test
`test_get_messages_orders_assistant_before_tool_rows_within_one_turn`
calls `persist_compose_turn` with a multi-tool turn on a fast SQLite
in-memory engine and asserts `get_messages` returns the assistant
row before every tool row of that turn. The test would have failed
on the pre-B2 codebase and passes after Change 1 lands.

---

## Task 15: Schema-level INV-AUDIT-AHEAD backward-direction integration test

**Files:**
- Create: `tests/integration/web/test_inv_audit_ahead_backward.py`

- [ ] **Step 1: Write the test**

Create `tests/integration/web/test_inv_audit_ahead_backward.py`:

```python
"""Spec §4.1.2 / §1.4 NFR: state-ahead-of-audit is impossible at the
schema level. After any persist_compose_turn call, the SQL predicate
below must return zero rows."""
from __future__ import annotations

import pytest
from sqlalchemy import text
from sqlalchemy.pool import StaticPool

from elspeth.web.sessions.service import SessionServiceImpl
from elspeth.web.sessions._persist_payload import _RedactedToolRow, _StatePayload
from elspeth.web.sessions.protocol import CompositionStateData
from elspeth.web.sessions.engine import create_session_engine
from elspeth.web.sessions.schema import initialize_session_schema
from elspeth.web.sessions.telemetry import build_sessions_telemetry
import structlog

# ``_make_session`` lives in ``tests/integration/web/conftest.py`` — a
# duplicate of the unit-test conftest helper. Importing the helper
# here keeps the per-test session-insert site uniform with the rest
# of the suite.
from .conftest import _make_session


@pytest.fixture
def service(tmp_path):
    """Service with an in-memory SQLite engine. The test runs
    end-to-end against the real production code paths
    (``create_session_engine`` + ``initialize_session_schema``);
    integration here means "exercises persist_compose_turn against
    a real SQLite engine," not "uses Docker" — see the conftest
    docstring for why integration and unit conftests are separate."""
    eng = create_session_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    initialize_session_schema(eng)
    return SessionServiceImpl(
        eng, data_dir=tmp_path,
        telemetry=build_sessions_telemetry(),
        log=structlog.get_logger(),
    )


_BACKWARD_PREDICATE = """
SELECT cs.id
  FROM composition_states cs
  LEFT JOIN chat_messages cm
    ON cm.composition_state_id = cs.id AND cm.role = 'tool'
 WHERE cs.provenance = 'tool_call' AND cs.version > 0
   AND cm.id IS NULL
"""


def test_backward_direction_holds_after_successful_persist(service):
    with service._engine.begin() as conn:
        _make_session(conn, session_id="b1")
    service.persist_compose_turn(
        session_id="b1",
        assistant_content="ok",
        redacted_assistant_tool_calls=({"id": "tc_a", "function": {"name": "f"}},),
        redacted_tool_rows=(
            _RedactedToolRow(
                "tc_a",
                '{"r": 1}',
                # B1 (Phase 1 plan-review synthesis): no ``version=``;
                # ``_insert_composition_state`` allocates under the lock.
                _StatePayload(
                    data=CompositionStateData(),
                    derived_from_state_id=None,
                ),
            ),
        ),
        parent_composition_state_id=None,
        expected_current_state_id=None,
        writer_principal="compose_loop",
        plugin_crash_pending=False,
    )
    with service._engine.begin() as conn:
        violations = conn.execute(text(_BACKWARD_PREDICATE)).fetchall()
    assert violations == [], f"backward-direction violation rows: {violations}"


def test_backward_direction_holds_after_integrity_error_rollback(service):
    """After a failed persist_compose_turn (rolled back transaction), no
    composition_states row should be visible."""
    from sqlalchemy.exc import IntegrityError

    with service._engine.begin() as conn:
        _make_session(conn, session_id="b2")
    # First successful turn.
    service.persist_compose_turn(
        session_id="b2",
        assistant_content="",
        redacted_assistant_tool_calls=({"id": "tc_x", "function": {"name": "f"}},),
        redacted_tool_rows=(
            _RedactedToolRow(
                "tc_x",
                "{}",
                # B1: no ``version=``; helper allocates under the lock.
                _StatePayload(
                    data=CompositionStateData(),
                    derived_from_state_id=None,
                ),
            ),
        ),
        parent_composition_state_id=None,
        expected_current_state_id=None,
        writer_principal="compose_loop",
        plugin_crash_pending=False,
    )
    with service._engine.begin() as conn:
        first_state_id = conn.execute(text(
            "SELECT id FROM composition_states "
            "WHERE session_id='b2' ORDER BY version DESC LIMIT 1"
        )).scalar_one()
    # Second turn deliberately reuses tc_x to trigger the partial
    # unique index ``uq_chat_messages_tool_call_id`` (added in Task 2).
    with pytest.raises(
        IntegrityError,
        match=(
            r"(UNIQUE.*chat_messages.*session_id.*tool_call_id"
            r"|uq_chat_messages_tool_call_id)"
        ),
    ):
        service.persist_compose_turn(
            session_id="b2",
            assistant_content="",
            redacted_assistant_tool_calls=({"id": "tc_x", "function": {"name": "f"}},),
            redacted_tool_rows=(
                _RedactedToolRow(
                    "tc_x",
                    "{}",
                    # B1: no ``version=``.
                    _StatePayload(
                        data=CompositionStateData(),
                        derived_from_state_id=None,
                    ),
                ),
            ),
            parent_composition_state_id=None,
            expected_current_state_id=first_state_id,
            writer_principal="compose_loop",
            plugin_crash_pending=False,
        )
    with service._engine.begin() as conn:
        violations = conn.execute(text(_BACKWARD_PREDICATE)).fetchall()
    assert violations == []
    # And exactly one tool_call provenance row from the first (successful) turn.
    state_count = conn.execute(text(
        "SELECT COUNT(*) AS c FROM composition_states "
        "WHERE session_id='b2' AND provenance='tool_call'"
    )).scalar()
    assert state_count == 1


def test_get_messages_orders_assistant_before_tool_rows_within_one_turn(service):
    """B2 (Phase 1 plan-review synthesis): a single ``persist_compose_turn``
    stamps every row in the turn with one shared ``created_at`` = ``now``;
    on fast SQLite the rows share a microsecond, so ``get_messages``'s
    pre-B2 ``ORDER BY created_at`` returned them nondeterministically.
    Post-B2 ``get_messages`` orders by ``sequence_no`` (allocated under
    the session write lock = monotonic and unique), so the
    intra-turn order is the order in which the writer appended rows
    (assistant first, tool rows in plan order). This test would have
    failed on the pre-B2 codebase.
    """
    from uuid import UUID

    # B2 (Phase 1 plan-review synthesis): pre-B2 this test bound
    # ``sid="ord1"`` and ``sid_uuid=UUID("00000000-...-001")``, then
    # inserted the session under ``sid`` and queried under
    # ``sid_uuid`` — two different sessions. The fix derives one
    # canonical UUID and uses its string form for ``_make_session`` /
    # ``persist_compose_turn`` and the UUID form for ``get_messages``.
    sid_uuid = UUID("00000000-0000-0000-0000-000000000001")
    sid = str(sid_uuid)
    with service._engine.begin() as conn:
        _make_session(conn, session_id=sid)
    # B2: ``assistant_id`` and ``assistant_raw_content`` were stale
    # kwargs from a prior plan draft that ``persist_compose_turn`` does
    # not declare. ``assistant_id`` is helper-generated (the test never
    # referenced the supplied value); ``assistant_raw_content`` was a
    # typo for the new ``raw_content`` parameter and is left at its
    # default ``None`` here because the test's narrative does not
    # exercise the redaction-attribution path.
    service.persist_compose_turn(
        session_id=sid,
        assistant_content="ok",
        redacted_assistant_tool_calls=(
            {"id": "tc_a", "function": {"name": "f"}},
            {"id": "tc_b", "function": {"name": "g"}},
            {"id": "tc_c", "function": {"name": "h"}},
        ),
        # B1 (Phase 1 plan-review synthesis): no ``version=`` kwargs.
        # ``_insert_composition_state`` allocates per-session contiguous
        # versions (1, 2, 3) under _session_write_lock.
        redacted_tool_rows=(
            _RedactedToolRow("tc_a", "{}", _StatePayload(data=CompositionStateData(), derived_from_state_id=None)),
            _RedactedToolRow("tc_b", "{}", _StatePayload(data=CompositionStateData(), derived_from_state_id=None)),
            _RedactedToolRow("tc_c", "{}", _StatePayload(data=CompositionStateData(), derived_from_state_id=None)),
        ),
        parent_composition_state_id=None,
        expected_current_state_id=None,
        writer_principal="compose_loop",
        plugin_crash_pending=False,
    )

    # ``get_messages`` is async (returns ChatMessageRecord objects); the
    # post-B2 ORDER BY sequence_no clause guarantees a stable order.
    import asyncio
    msgs = asyncio.run(service.get_messages(sid_uuid))
    roles = [m.role for m in msgs]
    # Exactly four rows from this turn: 1 assistant + 3 tool. No fork
    # rows, no system messages — _make_session left the chat empty.
    assert roles == ["assistant", "tool", "tool", "tool"], (
        f"intra-turn ordering broken: expected assistant before all tool rows, "
        f"got {roles!r} — see plan §14.7 (B2 fix). The pre-B2 ORDER BY created_at "
        f"would produce a nondeterministic permutation of these four roles."
    )
    # And the tool_call_id sequence is preserved (a→b→c, the order
    # ``redacted_tool_rows`` was supplied in).
    tool_ids = [m.tool_call_id for m in msgs if m.role == "tool"]
    assert tool_ids == ["tc_a", "tc_b", "tc_c"], (
        f"intra-tool ordering broken: {tool_ids!r}"
    )
```

- [ ] **Step 2: Run the test to verify it passes**

```bash
.venv/bin/python -m pytest tests/integration/web/test_inv_audit_ahead_backward.py -v
```
Expected: PASS — including the new
`test_get_messages_orders_assistant_before_tool_rows_within_one_turn`.

- [ ] **Step 3: Commit**

```bash
git add tests/integration/web/test_inv_audit_ahead_backward.py
git commit -m "test(integration): schema-level INV-AUDIT-AHEAD backward-direction + intra-turn ordering (composer-progress-persistence phase 1)"
```

---

## Task 16: CL-PP-11 concurrent multi-session test (testcontainer PostgreSQL)

**Files:**
- Create: `tests/integration/web/test_compose_loop_concurrent_sessions.py`
- Modify: `pyproject.toml` — register the `testcontainer` pytest marker AND add `testcontainers[postgres]` to the dev optional-dependencies group.
- Modify: `uv.lock` — required because CI uses frozen dependency sync.
- Modify: `.github/workflows/ci.yaml` — add a Docker-enabled testcontainer lane/step, update every explicit non-Docker pytest marker expression to exclude `testcontainer`, and include the Docker lane in the aggregate `ci-success` result check.

**Why a dedicated marker.** The default local test invocation
(`pytest tests/`) is configured with
`addopts = ["-m", "not slow and not stress and not performance"]`
in `pyproject.toml` and does NOT auto-skip Docker-required tests.
The GitHub Actions workflow also contains explicit `pytest -m "not slow
and not stress and not performance"` commands, so changing only
`pyproject.toml` is insufficient. Without registering a `testcontainer`
marker AND adding it to the default deselect list and every non-Docker
CI marker expression, every developer or CI job without Docker gets a
noisy fail. With the marker registered (and the default/CI non-Docker
selectors updated to deselect it) the test runs in CI's Docker-enabled
lane and is opt-in locally via `pytest -m testcontainer`. Closes
synthesised review findings M8 / Q-F-07.

- [ ] **Step 1: Register the `testcontainer` marker and add the dependency**

In `pyproject.toml`, extend the existing markers list under
`[tool.pytest.ini_options]` (around line 350) to include:

```toml
markers = [
    "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    "integration: marks tests requiring external services",
    "asyncio: marks tests as async (handled by pytest-asyncio)",
    "performance: marks tests as performance benchmarks",
    "stress: marks tests as stress tests requiring ChaosLLM HTTP server (deselect with '-m \"not stress\"')",
    "composer_llm_eval: characterization/replay tests for the 2026-04-28 composer LLM evaluation scenarios",
    "chaosllm: Configure ChaosLLM server for the test (preset=None, **kwargs)",
    "testcontainer: marks tests requiring Docker via testcontainers-python (deselect with '-m \"not testcontainer\"')",
]
```

In the same `[tool.pytest.ini_options]` block, extend the default
`addopts` deselect list:

```toml
addopts = [
    "-m", "not slow and not stress and not performance and not testcontainer",
]
```

This keeps the default `pytest` invocation Docker-free; CI's
Docker-enabled lane runs `pytest -m testcontainer` explicitly.

In the `[project.optional-dependencies]` table (the `dev` group),
add the testcontainers dependency:

```toml
dev = [
    # ...existing deps...
    "testcontainers[postgres]>=4.0,<5",
]
```

Run `uv sync --extra dev` after editing to install and update `uv.lock`.

- [ ] **Step 2: Confirm the install**

```bash
.venv/bin/python -c "import testcontainers.postgres" 2>&1 || echo "missing"
```
Expected: no output (success).

- [ ] **Step 2b: Add the CI lane**

In `.github/workflows/ci.yaml`, add a Docker-enabled job or explicit
step that installs the dev dependencies from the updated `uv.lock` and
runs:

```bash
.venv/bin/python -m pytest -m testcontainer tests/integration/web/ -v
```

Do not rely on the default unit/integration jobs: `addopts` deselects
`testcontainer` by design. The lane must fail if PostgreSQL cannot be
started through testcontainers. If this is implemented as a separate
GitHub Actions job, add that job name to the aggregate `ci-success`
job's `needs:` list AND update the `ci-success` shell/script logic that
checks `needs.<job>.result` so a failed or skipped testcontainer job
fails the aggregate gate; otherwise branch protection can remain green
while CL-PP-11 is red. If it is implemented as a step inside an existing
job that already feeds `ci-success`, update that job's explicit marker
expressions to:

```bash
-m "not slow and not stress and not performance and not testcontainer"
```

Do this for both coverage and non-coverage test commands. Document the
chosen CI shape in the PR body.

- [ ] **Step 3: Write the test**

Create `tests/integration/web/test_compose_loop_concurrent_sessions.py`:

```python
"""CL-PP-11 (spec §8.2): concurrent writes against two sessions that share
a PostgreSQL connection pool. Verifies (a) per-session sequence_no values
are independently monotonic, (b) advisory-lock collisions serialise but
do not deadlock, (c) the two-argument advisory-lock form
(``pg_advisory_xact_lock(ELSPETH_SESSIONS_LOCK_CLASSID,
hashtext(session_id))``) partitions ELSPETH's
session-write namespace away from any other application or
ELSPETH subsystem on the same Postgres cluster (B3 from the Phase 1
plan-review synthesis). ``hashtext`` returns the required int4 key
directly; do not replace it with ``hashtextextended(... )::int``
because PostgreSQL integer casts are range-checked. Within the
``ELSPETH_SESSIONS_LOCK_CLASSID`` namespace birthday collisions become
probable around ~65k *concurrent* sessions — collisions cause benign
extra serialisation only, never duplicate rows (the unique index
``ix_chat_messages_session_sequence`` is the correctness guarantee)."""
from __future__ import annotations

import threading
import uuid

import pytest
from sqlalchemy import text
from testcontainers.postgres import PostgresContainer

from elspeth.web.sessions.service import SessionServiceImpl
from elspeth.web.sessions._persist_payload import _RedactedToolRow, _StatePayload
from elspeth.web.sessions.protocol import CompositionStateData
from elspeth.web.sessions.engine import create_session_engine
from elspeth.web.sessions.schema import initialize_session_schema
from elspeth.web.sessions.telemetry import build_sessions_telemetry
import structlog

# Integration-suite shared session-insert helper.
from .conftest import _make_session


@pytest.fixture(scope="module")
def pg_engine():
    """Spin up a Postgres testcontainer for the duration of this test
    module. ``create_session_engine`` is a no-op for non-SQLite
    dialects (FK enforcement is the database's responsibility on
    Postgres) — using it here keeps the engine-construction pattern
    uniform with the rest of the suite. ``initialize_session_schema``
    is the same path production uses; we explicitly avoid
    ``metadata.create_all`` here because the production schema
    bootstrap is the only path tested in CI."""
    with PostgresContainer("postgres:16-alpine") as pg:
        engine = create_session_engine(pg.get_connection_url())
        initialize_session_schema(engine)
        yield engine


@pytest.fixture
def service(pg_engine, tmp_path):
    return SessionServiceImpl(
        pg_engine, data_dir=tmp_path,
        telemetry=build_sessions_telemetry(),
        log=structlog.get_logger(),
    )


import traceback


def _worker(
    service,
    session_id: str,
    writer_id: str,
    n: int,
    errors: list,
    tracebacks: list,
) -> None:
    """Concurrent-write worker. Captures the traceback alongside any
    exception so test failures are diagnosable rather than just
    showing the exception class. Closes synthesised review nit P-N-2."""
    try:
        for i in range(n):
            tool_call_id = f"{session_id}_{writer_id}_tc_{i}"
            service.persist_compose_turn(
                session_id=session_id,
                assistant_content=f"turn {i}",
                redacted_assistant_tool_calls=(
                    {"id": tool_call_id, "function": {"name": "f"}},
                ),
                redacted_tool_rows=(
                    _RedactedToolRow(
                        tool_call_id,
                        "{}",
                        None,
                    ),
                ),
                parent_composition_state_id=None,
                expected_current_state_id=None,
                writer_principal="compose_loop",
                plugin_crash_pending=False,
            )
    except Exception as exc:  # noqa: BLE001
        errors.append(exc)
        tracebacks.append(traceback.format_exc())


def test_concurrent_DIFFERENT_sessions_do_not_deadlock(service):
    """Concurrent writes to two DIFFERENT sessions must not interfere.
    Different ``session_id`` values map to different
    ``hashtext(...)`` slots within the
    ``ELSPETH_SESSIONS_LOCK_CLASSID`` namespace, so the advisory locks
    do NOT contend (modulo ~65k-concurrent-session birthday-collision
    probability — see module docstring) — this test verifies the
    *non-contending* path: two sessions can write concurrently with
    both reaching their expected row counts.

    For the advisory-lock CONTENTION path see
    ``test_concurrent_SAME_session_serialises_via_advisory_lock``
    below — that is the test the spec's CL-PP-11 description was
    actually targeting."""
    with service._engine.begin() as conn:
        _make_session(conn, session_id="s_a")
        _make_session(conn, session_id="s_b")

    errors: list[Exception] = []
    tracebacks: list[str] = []

    threads = [
        threading.Thread(target=_worker, args=(service, "s_a", "writer_a", 5, errors, tracebacks)),
        threading.Thread(target=_worker, args=(service, "s_b", "writer_b", 5, errors, tracebacks)),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    assert not errors, f"errors: {errors}\ntracebacks:\n" + "\n---\n".join(tracebacks)

    # Both sessions reached 10 messages each (1 assistant + 1 tool per turn × 5).
    with service._engine.begin() as conn:
        for sid in ("s_a", "s_b"):
            count = conn.execute(text(
                "SELECT COUNT(*) FROM chat_messages WHERE session_id = :sid"
            ), {"sid": sid}).scalar()
            assert count == 10, f"session {sid} count: {count}"
            seqs = [
                row.sequence_no
                for row in conn.execute(text(
                    "SELECT sequence_no FROM chat_messages "
                    "WHERE session_id = :sid ORDER BY sequence_no"
                ), {"sid": sid})
            ]
            # Strictly monotonic; gaps allowed but not duplicates.
            assert all(b > a for a, b in zip(seqs, seqs[1:]))


def test_concurrent_SAME_session_serialises_via_advisory_lock(service):
    """Concurrent writes to the SAME session must serialise via the
    session-scoped PostgreSQL advisory lock
    (``pg_advisory_xact_lock(ELSPETH_SESSIONS_LOCK_CLASSID,
    hashtext(session_id))`` — the two-argument form
    introduced by B3 to partition ELSPETH's namespace from other
    cluster users).

    Without serialisation, two writers would race on
    ``_reserve_sequence_range``'s ``SELECT MAX(sequence_no) ...``
    and produce duplicate ``sequence_no`` values, hitting the
    ``ix_chat_messages_session_sequence`` unique index and raising
    ``IntegrityError``. With the advisory lock, the second writer
    blocks until the first commits, sees the updated max, and
    allocates the next contiguous range — no duplicates, no
    deadlock.

    Closes synthesised review finding F-02 / Q-F-02 (CL-PP-11
    coverage gap)."""
    with service._engine.begin() as conn:
        _make_session(conn, session_id="s_shared")

    errors: list[Exception] = []
    tracebacks: list[str] = []

    threads = [
        threading.Thread(
            target=_worker,
            args=(service, "s_shared", "writer_A", 10, errors, tracebacks),
            name="writer_A",
        ),
        threading.Thread(
            target=_worker,
            args=(service, "s_shared", "writer_B", 10, errors, tracebacks),
            name="writer_B",
        ),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    assert not errors, (
        f"advisory-lock serialisation failed; errors:\n"
        f"{errors}\ntracebacks:\n" + "\n---\n".join(tracebacks)
    )

    # Both writers must have completed; total rows = 2 × 10 turns ×
    # 2 rows/turn (one assistant + one tool) = 40.
    with service._engine.begin() as conn:
        count = conn.execute(text(
            "SELECT COUNT(*) FROM chat_messages WHERE session_id = 's_shared'"
        )).scalar()
        assert count == 40, f"expected 40 rows, got {count}"

        # sequence_no values must be unique across both writers.
        # Duplicates would manifest as IntegrityError above; the
        # explicit check here documents the expected post-condition
        # so a future regression that disables the unique index
        # (instead of the lock) still fails the test.
        seqs = [
            row.sequence_no
            for row in conn.execute(text(
                "SELECT sequence_no FROM chat_messages "
                "WHERE session_id = 's_shared' ORDER BY sequence_no"
            ))
        ]
        assert len(seqs) == len(set(seqs)), (
            f"duplicate sequence_no values: "
            f"{[s for s in seqs if seqs.count(s) > 1]}"
        )
        # Strict monotonicity: 1..40 with no gaps under successful
        # serial allocation. (Gaps are allowed in general — see spec
        # §4.1.1 — but happen only on rollback paths, none of which
        # this test exercises.)
        assert seqs == list(range(min(seqs), min(seqs) + 40))


def test_advisory_lock_actually_acquired_on_postgres(service):
    """Positive assertion against ``pg_locks``: when
    ``_acquire_session_advisory_lock`` is called inside an open
    transaction on PostgreSQL, a row appears in ``pg_locks`` with
    ``locktype='advisory'``, ``granted=true``,
    ``classid=ELSPETH_SESSIONS_LOCK_CLASSID``, and
    ``objid=hashtext('lock_probe')``. Without this,
    a typo in the dialect guard (e.g. checking ``"postgres"``
    instead of ``"postgresql"``) would silently no-op the lock
    on PG and the contention test above would still pass for
    the wrong reason — the SELECT MAX races would simply not
    overlap in test time.

    Filtering on the classid is load-bearing for B3 (Phase 1
    plan-review synthesis): if a future second classid lands in the
    same cluster, an unfiltered ``classid=*`` assertion would pass
    against the wrong subsystem's lock and silently green-light a
    regression in the sessions-DB lock path. The classid is the
    namespace-partitioning guarantee the two-argument advisory-lock
    form was introduced to provide.

    Closes synthesised review finding F-03 / Q-F-03."""
    from elspeth.contracts.advisory_locks import ELSPETH_SESSIONS_LOCK_CLASSID

    with service._engine.begin() as conn:
        _make_session(conn, session_id="lock_probe")
        # Acquire the advisory lock; do NOT release manually —
        # ``pg_advisory_xact_lock`` releases at transaction end.
        service._acquire_session_advisory_lock(conn, "lock_probe")
        # Inspect pg_locks while still inside the transaction.
        # Filter on classid AND objid so this test fails loudly if a
        # future change accidentally drops the classid (collapsing
        # back to single-argument form) or hashes the wrong session_id.
        row = conn.execute(text(
            "SELECT classid, objid FROM pg_locks "
            "WHERE locktype = 'advisory' AND granted = true "
            "AND classid = :classid"
        ), {"classid": ELSPETH_SESSIONS_LOCK_CLASSID}).first()
        assert row is not None, (
            "advisory lock not visible in pg_locks under "
            f"classid={ELSPETH_SESSIONS_LOCK_CLASSID} — dialect guard "
            "in _acquire_session_advisory_lock may have skipped the "
            "Postgres branch, or the two-argument form was reverted "
            "to single-argument (regressing B3)."
        )
        # Verify objid matches hashtext(session_id).
        # Computing the expected value via the database guarantees we
        # exercise the same hash function the helper used.
        expected_objid = conn.execute(text(
            "SELECT hashtext(:sid)"
        ), {"sid": "lock_probe"}).scalar()
        assert row.objid == expected_objid, (
            f"advisory-lock objid mismatch: got {row.objid}, "
            f"expected {expected_objid} — _acquire_session_advisory_lock "
            "may be hashing a different value than session_id."
        )


def test_cross_allocator_save_state_serialises_with_persist_turn(service):
    """B3 (Phase 1 plan-review synthesis): the dual-allocator race.

    Pre-B3, ``persist_compose_turn`` allocated ``composition_states.version``
    under _session_write_lock, but ``save_composition_state``
    relied on a 3-attempt retry loop with NO lock. A concurrent
    interleave produced a real ``IntegrityError`` on the locked path,
    which the locked path's exception handler classified as a Tier-1
    audit-integrity violation — fabricating a Tier-1 violation from a
    benign race. (Fabricated Tier-1 violations are evidence-tampering
    under the audit-grade doctrine.)

    Post-B3, ``save_composition_state`` (and ``set_active_state``)
    also enter ``_session_write_lock`` before their SELECT MAX. The
    persist side also has the stale-current-state guard introduced in
    Task 11: if a save wins after the persist worker reads the current
    state but before it acquires the write lock, ``persist_compose_turn``
    rejects with ``StaleComposeStateError``. That is the correct
    compose-vs-revert disposition and must NOT be counted as an audit
    integrity failure. Successful writers must still leave sequential,
    contiguous version numbers in ``composition_states`` with NO
    ``IntegrityError`` raised on either path.

    This test would have failed pre-B3 by either crashing one path
    with ``IntegrityError`` or, worse, by silently incrementing the
    ``tool_row_integrity_violation_total`` counter (the false-positive
    Tier-1 alert) on the persist path.
    """
    from elspeth.web.sessions._persist_payload import _RedactedToolRow, _StatePayload
    from elspeth.web.sessions.protocol import CompositionStateData
    from elspeth.web.sessions.service import StaleComposeStateError
    from elspeth.web.sessions.telemetry import observed_value

    session_uuid = uuid.uuid4()
    sid = str(session_uuid)
    with service._engine.begin() as conn:
        _make_session(conn, session_id=sid)

    errors: list[Exception] = []
    stale_rejections: list[StaleComposeStateError] = []
    tracebacks: list[str] = []
    persist_count = 5
    save_count = 5
    starting_counter = observed_value(
        service._telemetry.tool_row_integrity_violation_total
    )

    def _persist_worker():
        # B2 (Phase 1 plan-review synthesis): pre-B2 this loop passed
        # ``assistant_id=str(uuid.uuid4())`` and
        # ``assistant_raw_content=None`` to ``persist_compose_turn`` —
        # neither kwarg is declared on the signature.
        # ``assistant_id`` is helper-generated (the test never read it
        # back); ``assistant_raw_content`` was a typo for the new
        # ``raw_content`` parameter, which defaults to ``None`` and so
        # is omitted entirely (this test exercises the
        # cross-allocator race, not the redaction-attribution path).
        try:
            for i in range(persist_count):
                with service._engine.begin() as conn:
                    expected_current_state_id = conn.execute(text(
                        "SELECT id FROM composition_states "
                        "WHERE session_id = :sid ORDER BY version DESC LIMIT 1"
                    ), {"sid": sid}).scalar_one_or_none()
                tool_call_id = f"tc_persist_{i}"
                service.persist_compose_turn(
                    session_id=sid,
                    assistant_content="ok",
                    redacted_assistant_tool_calls=({"id": tool_call_id, "function": {"name": "f"}},),
                    redacted_tool_rows=(
                        # B1: no ``version=``; helper allocates under lock.
                        _RedactedToolRow(tool_call_id, "{}", _StatePayload(data=CompositionStateData(), derived_from_state_id=None)),
                    ),
                    parent_composition_state_id=None,
                    expected_current_state_id=expected_current_state_id,
                    writer_principal="compose_loop",
                    plugin_crash_pending=False,
                )
        except StaleComposeStateError as e:
            stale_rejections.append(e)
        except Exception as e:
            import traceback
            errors.append(e)
            tracebacks.append(traceback.format_exc())

    def _save_worker():
        import asyncio
        try:
            for _ in range(save_count):
                asyncio.run(
                    service.save_composition_state(
                        session_uuid,  # ``save_composition_state`` takes UUID
                        CompositionStateData(),
                    )
                )
        except Exception as e:
            import traceback
            errors.append(e)
            tracebacks.append(traceback.format_exc())

    threads = [
        threading.Thread(target=_persist_worker, name="persist"),
        threading.Thread(target=_save_worker, name="save"),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    assert not errors, (
        f"cross-allocator race not serialised; errors:\n"
        f"{errors}\ntracebacks:\n" + "\n---\n".join(tracebacks)
    )
    assert observed_value(
        service._telemetry.tool_row_integrity_violation_total
    ) == starting_counter

    # Sanity: every allocated version is unique and contiguous from 1.
    with service._engine.begin() as conn:
        versions = sorted(
            row.version
            for row in conn.execute(text(
                "SELECT version FROM composition_states "
                "WHERE session_id = :sid ORDER BY version"
            ), {"sid": sid})
        )
    # Successful writers only: stale compose attempts roll back before
    # inserting state rows. Whatever succeeded must still be contiguous
    # from 1 with no duplicate/gap evidence of allocator races.
    assert versions == list(range(1, len(versions) + 1)), (
        f"cross-allocator versions not contiguous: {versions}"
    )
    assert len(versions) + len(stale_rejections) >= save_count


def test_concurrent_persist_compose_turn_same_state_stale_rejects_without_integrity_alert(service):
    """B1 plus stale-state guard: two concurrent state-changing compose
    persists based on the same current state must not fabricate a Tier-1
    integrity alert.

    Pre-B1, both writers could try to insert the same caller-supplied
    ``composition_states.version`` and one would hit
    ``uq_composition_state_version``. Post-B1 the helper allocates the
    version under the lock, but Task 11's stale-current-state guard adds
    the stronger user-intent rule: after the first writer creates the new
    current state, the second writer's ``expected_current_state_id=None``
    is stale and should be rejected with ``StaleComposeStateError`` before
    it inserts any audit rows. The integrity counter must not move.

    Closes B1 and the compose-vs-revert stale-state race from the Phase 1
    plan-review synthesis."""
    from elspeth.web.sessions._persist_payload import _RedactedToolRow, _StatePayload
    from elspeth.web.sessions.protocol import CompositionStateData
    from elspeth.web.sessions.service import StaleComposeStateError
    from elspeth.web.sessions.telemetry import observed_value

    sid = "s_b1_concurrent"
    with service._engine.begin() as conn:
        _make_session(conn, session_id=sid)

    starting_counter = observed_value(
        service._telemetry.tool_row_integrity_violation_total
    )

    errors: list[Exception] = []
    stale_rejections: list[StaleComposeStateError] = []
    tracebacks: list[str] = []
    barrier = threading.Barrier(2)

    def _stateful_worker(writer_id: str) -> None:
        try:
            barrier.wait(timeout=10)
            tool_call_id = f"{sid}_{writer_id}_tc_0"
            service.persist_compose_turn(
                session_id=sid,
                assistant_content="ok",
                redacted_assistant_tool_calls=(
                    {"id": tool_call_id, "function": {"name": "f"}},
                ),
                redacted_tool_rows=(
                    _RedactedToolRow(
                        tool_call_id,
                        "{}",
                        _StatePayload(
                            data=CompositionStateData(),
                            derived_from_state_id=None,
                        ),
                    ),
                ),
                parent_composition_state_id=None,
                expected_current_state_id=None,
                writer_principal="compose_loop",
                plugin_crash_pending=False,
            )
        except StaleComposeStateError as exc:
            stale_rejections.append(exc)
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)
            tracebacks.append(traceback.format_exc())

    threads = [
        threading.Thread(
            target=_stateful_worker,
            args=("b1_writer_A",),
            name="b1_writer_A",
        ),
        threading.Thread(
            target=_stateful_worker,
            args=("b1_writer_B",),
            name="b1_writer_B",
        ),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    # B1/stale-guard: NO IntegrityError on either path. The loser of the
    # same-state race is a stale compose, not an audit-integrity event.
    assert not errors, (
        f"B1 regression: dual-allocator race not closed. errors:\n"
        f"{errors}\ntracebacks:\n" + "\n---\n".join(tracebacks)
    )
    assert len(stale_rejections) == 1

    # B1: the integrity-violation counter MUST NOT have moved. Any
    # increment here is a fabricated Tier-1 alert (SLO threshold = 0).
    assert observed_value(
        service._telemetry.tool_row_integrity_violation_total
    ) == starting_counter, (
        "B1 regression: tool_row_integrity_violation_total incremented "
        "during concurrent persist_compose_turn. SLO threshold for this "
        "counter is 0; any increment is a fabricated Tier-1 alert and "
        "evidence-tampering-class harm under ELSPETH's audit doctrine."
    )

    # Exactly one writer commits; the stale writer rolls back before
    # inserting any state or chat rows. The one committed state receives
    # version 1.
    with service._engine.begin() as conn:
        versions = [
            row.version
            for row in conn.execute(text(
                "SELECT version FROM composition_states "
                "WHERE session_id = :sid ORDER BY version"
            ), {"sid": sid})
        ]
    assert versions == [1]
```

- [ ] **Step 4: Apply the `testcontainer` marker to the whole module**

The test file uses free functions, not a test class. Add module-level
`pytestmark` immediately after imports so every function in
`tests/integration/web/test_compose_loop_concurrent_sessions.py` is
selected by `pytest -m testcontainer` and deselected by the default
Docker-free lane:

```python
pytestmark = pytest.mark.testcontainer
```

This both signals intent to readers and (with the
`addopts = ["-m", "not ... and not testcontainer"]` deselect from
Step 1) keeps the default `pytest tests/` invocation Docker-free.
CI's Docker-enabled lane runs `pytest -m testcontainer` to opt in.

- [ ] **Step 5: Run the test in the Docker-enabled lane**

```bash
.venv/bin/python -m pytest tests/integration/web/test_compose_loop_concurrent_sessions.py -v -m testcontainer
```
Expected: PASS. Verify locally only when Docker is running; in CI,
the testcontainer lane is the canonical execution path.

- [ ] **Step 6: Commit**

```bash
git add tests/integration/web/test_compose_loop_concurrent_sessions.py pyproject.toml uv.lock .github/workflows/ci.yaml
git commit -m "test(integration): CL-PP-11 concurrent multi-session writes against testcontainer Postgres (composer-progress-persistence phase 1)"
```

---

## Task 17: Latency sanity bound test

**Files:**
- Create: `tests/integration/web/test_compose_loop_latency_sanity.py`

- [ ] **Step 1: Write the test**

```python
"""Spec §1.4 NFR: per-turn DB write overhead p95 ≤ 250 ms with N ≤ 8 tool
calls, measured in CI on standard infra. Order-of-magnitude bound, not a
tight budget; the tight 25 ms target is verified by a nightly bench job."""
from __future__ import annotations

import statistics
import time

import pytest
from sqlalchemy.pool import StaticPool

from elspeth.web.sessions.service import SessionServiceImpl
from elspeth.web.sessions._persist_payload import _RedactedToolRow, _StatePayload
from elspeth.web.sessions.protocol import CompositionStateData
from elspeth.web.sessions.engine import create_session_engine
from elspeth.web.sessions.schema import initialize_session_schema
from elspeth.web.sessions.telemetry import build_sessions_telemetry
import structlog

# Integration-suite shared session-insert helper.
from .conftest import _make_session


@pytest.fixture
def service(tmp_path):
    eng = create_session_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    initialize_session_schema(eng)
    return SessionServiceImpl(
        eng, data_dir=tmp_path,
        telemetry=build_sessions_telemetry(),
        log=structlog.get_logger(),
    )


_WARMUP_TURNS = 5
_MEASURED_TURNS = 50
_TOTAL_TURNS = _WARMUP_TURNS + _MEASURED_TURNS


def _build_turn(turn: int):
    """Construct a (calls, rows) pair for one turn of the latency test.
    Extracted so the warm-up and measured loops share identical
    construction, removing one variable from the timing comparison."""
    rows = tuple(
        _RedactedToolRow(
            f"turn_{turn}_tc_{i}",
            '{"ok": true}',
            _StatePayload(
                data=CompositionStateData(),
                derived_from_state_id=None,
            ),
        )
        for i in range(8)
    )
    calls = tuple(
        {"id": f"turn_{turn}_tc_{i}", "function": {"name": "f"}}
        for i in range(8)
    )
    return calls, rows


def test_per_turn_p95_under_250ms_with_8_tool_calls(service):
    """Sanity bound (spec §1.4 NFR). Excludes the first ``_WARMUP_TURNS``
    iterations from the measurement: SQLAlchemy connection-pool warmup,
    Python bytecode caching, SQLite page-cache fill, and OTel meter
    materialisation all skew the very first calls. Without this
    exclusion the p95 absorbs cold-start noise that does not represent
    steady-state production latency. Closes synthesised review
    finding F-09 / Q-F-09.

    The bound is a sanity threshold, not a tight target — a tight
    25ms target is verified by a nightly bench job (see overview).

    **Scope note (synthesised review S6 / M3 — systems thinker).**
    This test measures the SYNC-ONLY path:
    ``service.persist_compose_turn(...)`` invoked directly. In
    production (Phase 3), the compose loop calls
    ``await service.persist_compose_turn_async(...)``, which
    additionally pays for ``ThreadPoolExecutor`` allocation + the
    ``run_in_executor`` scheduling round-trip. Phase 3 must add a
    parallel test that exercises the full async dispatch path so
    the production p95 — not just the sync function — is bounded.
    For Phase 1 the sync baseline is the right scope: Phase 1's
    only caller of ``persist_compose_turn`` is this test.
    """
    with service._engine.begin() as conn:
        _make_session(conn, session_id="lat")

    expected_current_state_id = None

    # Warm-up: discard timings.
    for turn in range(_WARMUP_TURNS):
        calls, rows = _build_turn(turn)
        service.persist_compose_turn(
            session_id="lat",
            assistant_content="",
            redacted_assistant_tool_calls=calls,
            redacted_tool_rows=rows,
            parent_composition_state_id=None,
            expected_current_state_id=expected_current_state_id,
            writer_principal="compose_loop",
            plugin_crash_pending=False,
        )
        with service._engine.begin() as conn:
            expected_current_state_id = conn.execute(text(
                "SELECT id FROM composition_states "
                "WHERE session_id='lat' ORDER BY version DESC LIMIT 1"
            )).scalar_one()

    # Measured: 50 turns is enough for stable p95 via 20-quantiles
    # (the 19th quantile == p95).
    durations: list[float] = []
    for turn in range(_WARMUP_TURNS, _TOTAL_TURNS):
        calls, rows = _build_turn(turn)
        start = time.perf_counter()
        service.persist_compose_turn(
            session_id="lat",
            assistant_content="",
            redacted_assistant_tool_calls=calls,
            redacted_tool_rows=rows,
            parent_composition_state_id=None,
            expected_current_state_id=expected_current_state_id,
            writer_principal="compose_loop",
            plugin_crash_pending=False,
        )
        durations.append((time.perf_counter() - start) * 1000)
        with service._engine.begin() as conn:
            expected_current_state_id = conn.execute(text(
                "SELECT id FROM composition_states "
                "WHERE session_id='lat' ORDER BY version DESC LIMIT 1"
            )).scalar_one()

    p95 = statistics.quantiles(durations, n=20)[18]  # 19th of 20 quantiles == p95
    assert p95 < 250, (
        f"p95={p95:.1f}ms exceeds 250ms sanity bound (warmup={_WARMUP_TURNS} "
        f"discarded, measured={_MEASURED_TURNS}); durations={durations}"
    )
```

- [ ] **Step 2: Run the test**

```bash
.venv/bin/python -m pytest tests/integration/web/test_compose_loop_latency_sanity.py -v
```
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/integration/web/test_compose_loop_latency_sanity.py
git commit -m "test(integration): per-turn latency sanity bound (composer-progress-persistence phase 1)"
```

---

## Task 18: Document the staging session-DB recreation procedure

**Why this task exists.** Phase 1 changes the `chat_messages` and
`composition_states` schemas. ELSPETH has no Alembic migration
framework for the web session DB, and
`src/elspeth/web/sessions/schema.py` validates existing schema shape
instead of altering tables in place. Row-level `DELETE FROM
chat_messages` / `DELETE FROM composition_states` is therefore
incorrect: it leaves the old table shape behind and startup rejects the
stale DB.

The correct pre-release procedure is to stop the service, archive the
current session DB file, remove the live file, and restart so
`initialize_session_schema` recreates the schema from current metadata.

**Files:**
- Create or modify: `docs/runbooks/staging-session-db-recreation.md`

- [ ] **Step 1: Create the runbook under the existing runbook tree**

Use `docs/runbooks/`, not a new `docs/operations/` directory. If there
is an index for runbooks, add this file there too.

- [ ] **Step 2: Add the session-DB recreation procedure**

Create `docs/runbooks/staging-session-db-recreation.md` with this
section. If any existing staging note references row-level DELETE SQL or
`elspeth migrate up` for the session DB, replace it.

````markdown
# Staging Session DB Recreation

ELSPETH staging (`elspeth.foundryside.dev`) serves the source checkout at
`/home/john/elspeth` through `elspeth-web.service`. The web session DB
has no Alembic migrations. When a pre-release plan changes the session
schema, recreate the DB file from current metadata.

This procedure destroys staging chat history, composition states, runs,
run events, and blob/blob-link records. Do not run it outside staging.

## Preconditions

1. The host is the staging host for `elspeth.foundryside.dev`.
2. No human operator is mid-session.
3. The source checkout at `/home/john/elspeth` is on the commit being
   deployed.
4. `deploy/elspeth-web.env` has been inspected directly for session DB
   settings without printing secret values.

## Procedure

```bash
set -euo pipefail

PROJECT_ROOT="/home/john/elspeth"
ENV_FILE="$PROJECT_ROOT/deploy/elspeth-web.env"
SERVICE="elspeth-web.service"

if [ "$(pwd)" != "$PROJECT_ROOT" ]; then
    echo "REFUSING: run from $PROJECT_ROOT so relative defaults are unambiguous." >&2
    exit 1
fi

if ! systemctl show "$SERVICE" --property=FragmentPath --value | grep -q "elspeth-web.service"; then
    echo "REFUSING: $SERVICE is not the active staging service." >&2
    exit 1
fi

# Resolve the DB path without echoing the environment file. Secrets may
# live in the same file. Precedence:
#   1. ELSPETH_WEB__SESSION_DB_URL=file-or-sqlite-url
#   2. ELSPETH_WEB__DATA_DIR/sessions.db
#   3. /home/john/elspeth/data/sessions.db
SESSION_DB_URL="$(grep -E '^ELSPETH_WEB__SESSION_DB_URL=' "$ENV_FILE" | tail -n1 | cut -d= -f2- || true)"
DATA_DIR="$(grep -E '^ELSPETH_WEB__DATA_DIR=' "$ENV_FILE" | tail -n1 | cut -d= -f2- || true)"

if [ -n "$SESSION_DB_URL" ]; then
    case "$SESSION_DB_URL" in
        sqlite:///*) DB_PATH="${SESSION_DB_URL#sqlite:///}" ;;
        sqlite:////*) DB_PATH="/${SESSION_DB_URL#sqlite:////}" ;;
        *) echo "REFUSING: non-sqlite session DB URL requires a migration plan." >&2; exit 1 ;;
    esac
elif [ -n "$DATA_DIR" ]; then
    DB_PATH="$DATA_DIR/sessions.db"
else
    DB_PATH="$PROJECT_ROOT/data/sessions.db"
fi

case "$DB_PATH" in
    "$PROJECT_ROOT"/*) ;;
    *) echo "REFUSING: DB_PATH is outside $PROJECT_ROOT: $DB_PATH" >&2; exit 1 ;;
esac

echo "Resolved staging session DB path: $DB_PATH"
read -r -p "Archive and recreate this staging DB? Type RECREATE to continue: " CONFIRM
if [ "$CONFIRM" != "RECREATE" ]; then
    echo "Aborted."
    exit 1
fi

sudo systemctl stop "$SERVICE"

if [ -e "$DB_PATH" ]; then
    SNAPSHOT="$DB_PATH.pre-phase1.$(date -u +%Y%m%dT%H%M%SZ)"
    sudo cp -a "$DB_PATH" "$SNAPSHOT"
    echo "Archived existing DB to $SNAPSHOT"
fi

sudo rm -f "$DB_PATH"
sudo systemctl start "$SERVICE"

curl --unix-socket /run/elspeth/uvicorn.sock -fsS http://localhost/api/health
curl -fsS https://elspeth.foundryside.dev/api/health
sudo systemctl status "$SERVICE" --no-pager --lines=20
```

`initialize_session_schema()` recreates the file on service startup.
If either health check fails, inspect `journalctl -u elspeth-web.service
--no-pager -n 80` before retrying.

## Why DELETE Is Forbidden

Row deletion does not add columns, CHECK constraints, FKs, or indexes.
For schema-changing Phase plans, row deletion leaves a stale DB shape
that the startup schema validator correctly rejects. Archive/delete the
file instead.
````

- [ ] **Step 3: Verify the procedure locally before committing**

Do a dry run against a throwaway copy of `data/sessions.db` or a
temporary local config path. Do not run the destructive staging command
from the Codex sandbox unless host/systemd access is explicitly
available. Record in the Phase 1 PR description:

- resolved DB path logic tested for default `data/sessions.db`
- archive path format
- health-check command used for local smoke

- [ ] **Step 4: Commit**

```bash
git add docs/runbooks/staging-session-db-recreation.md
git commit -m "docs(runbooks): session-DB recreation procedure for staging schema changes (composer-progress-persistence phase 1)"
```

---

## Task 19: File OQ-1 follow-up + amend stale design/spec handoff text

- [ ] **Step 1: Use the filigree MCP tool to create the OQ-1 issue**

Create a new issue with the following fields. Use the `mcp__filigree__create_issue` tool, or `filigree create` CLI if MCP is unavailable:

- Title: `chat_messages and audit_access_log retention CLI extension`
- Type: `task`
- Priority: `P3`
- Description: `Spec section: docs/superpowers/specs/2026-04-30-composer-progress-persistence-design.md §10 OQ-1. Extend the elspeth purge --retention-days CLI to operate on web session chat_messages and audit_access_log tables in addition to its existing pipeline-payload scope. Currently web tables grow only through cascade-delete with sessions; an explicit retention path is needed before the production composer corpus exists.`
- Labels: `cluster:composer-progress-persistence`, `from-design-spec`

- [ ] **Step 2: Record the OQ-1 ticket ID**

Record the assigned `elspeth-XXXXXXXX` ID. This will be cited in the Phase 1 PR description as the resolution to OQ-1.

- [ ] **Step 3: Amend the design/spec document in the same PR**

Modify `docs/superpowers/specs/2026-04-30-composer-progress-persistence-design.md`
so downstream phases do not copy stale source-of-truth text:

- §4.1.1: extend the `chat_messages.role` values with internal
  `"audit"` and extend `writer_principal` with `"session_fork"`.
  Explain that `role="audit"` rows are excluded from normal chat
  history and are used only for audit breadcrumbs that cannot be
  parented to an assistant row.
- §4.1.1: replace any one-column `parent_assistant_id` FK description
  with the composite same-session FK:
  `(parent_assistant_id, session_id) -> (chat_messages.id, chat_messages.session_id)`,
  backed by `uq_chat_messages_id_session`.
- §5.2 / §5.2.1: remove `version` from `_StatePayload` construction
  and field lists. State versions are allocated by
  `_insert_composition_state` under `_session_write_lock`.
- §5.2.2: add `raw_content: str | None = None` to
  `persist_compose_turn` and plumb it to the assistant
  `_insert_chat_message` call.
- §5.2.2: add `expected_current_state_id: str | None` to
  `persist_compose_turn` and document the under-lock stale-state guard:
  current latest state must equal the expected state or the helper raises
  `StaleComposeStateError` before sequence/version allocation.
- §5.2.2 / protocol handoff: public async callers use
  `SessionServiceProtocol.persist_compose_turn_async(...)`; the sync
  `SessionServiceImpl.persist_compose_turn(...)` stays concrete-only and
  async-loop guarded.
- §5.2.2 helper snippets: replace local `_enveloped(...)` assumptions
  with the shared `_enveloped_state_column(...)` helper used by
  `_insert_composition_state`, `save_composition_state`, and `fork_session`.
- §5.2.2: update `_AuditOutcome` to the two-field shape
  `(assistant_id, unwind_audit_failed)` and make Tier-1 audit-write
  failures raise `AuditIntegrityError` inside the sync worker rather
  than returning a flag for the caller to re-raise.
- §6.3: define `audit_access_log.session_id` with
  `ForeignKey("sessions.id", ondelete="CASCADE")` (or the equivalent SQL
  `ON DELETE CASCADE`) so `archive_session` can delete sessions that have
  audit-grade view rows.
- §5.7.1: replace the single-argument advisory-lock SQL with
  `_session_write_lock`: PostgreSQL uses
  `pg_advisory_xact_lock(ELSPETH_SESSIONS_LOCK_CLASSID,
  hashtext(session_id))` and SQLite uses a process-wide `(database_url,
  session_id)` RLock for the current single-process staging deployment.
- §8.6: replace bare `sqlalchemy.create_engine("sqlite:///:memory:")`
  + `metadata.create_all()` test guidance with
  `create_session_engine(..., poolclass=StaticPool)` +
  `initialize_session_schema()`.
- §10 / OQ-4: replace row-level DELETE framing with the
  session-DB archive/delete/restart recreation procedure.
- CI handoff text: document the `testcontainer` marker, explicit
  non-Docker CI marker expressions, and `ci-success` `needs.<job>.result`
  check for the Docker-enabled lane.

- [ ] **Step 4: Update overview handoff text if needed**

If this Phase 1 plan and the overview are in the same PR, confirm
`2026-04-30-composer-progress-persistence-overview.md` contains the
same supersession language: no row DELETE, no bare `create_engine`,
Phase 3 owns recovery fidelity, and Phase 3 cannot ship without
Tier-1 alert/dashboard/runbook visibility.

- [ ] **Step 5: Commit**

```bash
git add docs/superpowers/specs/2026-04-30-composer-progress-persistence-design.md \
        docs/superpowers/plans/completed/composer-progress-persistence/2026-04-30-composer-progress-persistence-overview.md
git commit -m "docs(composer): align progress-persistence spec with phase 1 execution plan"
```

- [ ] **Step 6: Record OQ-1 in the PR description**

The OQ-1 issue lives in Filigree, not the repo. Replace the
`elspeth-XXXXXXXX` placeholder in the PR body with the actual issue ID.
No `YYYYYYYY` observation placeholders should remain; spec drift is
handled by the doc amendments above, not by observation-only notes.

---

## Task 20: Final Schedule C / Phase 1 closure CI run

- [ ] **Step 1: Run the full sessions test suite**

```bash
.venv/bin/python -m pytest tests/unit/web/sessions/ tests/unit/web/sessions/test_telemetry.py tests/unit/web/composer/test_audit_failure_primacy.py tests/integration/web/test_inv_audit_ahead_backward.py tests/integration/web/test_compose_loop_latency_sanity.py -v
```
Expected: PASS for every test.

- [ ] **Step 1b: Run the testcontainer-marked tests in a Docker-enabled lane (closes B4)**

This step is required before opening the PR. Without it, the
"all CL-PP-* scenarios pass" done-condition (see "Phase 1 Done When"
below) is not actually verified — Task 16's `addopts = ["-m", "not
... and not testcontainer"]` deselect causes Step 2's `pytest tests/ -x`
to silently skip CL-PP-11 and the B3 cross-allocator test. The
synthesised plan-review (B4) ranked this gap as a blocker because
the headline concurrency contract ships unverified.

```bash
.venv/bin/python -m pytest -m testcontainer tests/integration/web/ -v
```
Expected: PASS. Requires Docker on the local machine OR running this
step in CI's Docker-enabled lane.

**If Docker is unavailable locally**, do NOT skip this step. Mark
the PR description with "[ ] CL-PP-11 + B3 cross-allocator pending
CI testcontainer lane" and confirm the testcontainer lane completes
green before merging. The Phase 1 sign-off explicitly depends on
Docker-lane CI green for these tests.

**Sanity check on test discovery** (catches the
"addopts deselected the file before pytest could collect it" failure
mode that B4 was filed against):

```bash
.venv/bin/python -m pytest -m testcontainer tests/integration/web/ --collect-only -q | grep -E "test_concurrent|test_advisory_lock_actually|test_cross_allocator"
```
Expected: at minimum the five CL-PP-11 / B3 / B1 test functions appear
in the collection output (`test_concurrent_DIFFERENT_sessions_do_not_deadlock`,
`test_concurrent_SAME_session_serialises_via_advisory_lock`,
`test_advisory_lock_actually_acquired_on_postgres`,
`test_cross_allocator_save_state_serialises_with_persist_turn`, and
`test_concurrent_persist_compose_turn_same_state_stale_rejects_without_integrity_alert`
— the B1/stale regression for the dual-``persist_compose_turn`` race
that the version-on-payload contract previously fabricated as a Tier-1
violation). If the grep returns empty, the marker registration or
`addopts` deselect is broken and the testcontainer lane is silently
green for the wrong reason.

- [ ] **Step 2: Run the full project test suite**

```bash
.venv/bin/python -m pytest tests/ -x
```
Expected: PASS. No regression in pre-existing tests. The testcontainer
tests are deselected here by design (Step 1b is the canonical run for
those); Step 2 is the Docker-free regression check.

- [ ] **Step 3: Run mypy**

```bash
uv run mypy src/ tests/
```
Expected: clean, matching the CI mypy scope. Address any new errors
before opening the PR.

- [ ] **Step 4: Run ruff**

```bash
uv run ruff check src/ tests/ scripts/ examples/
uv run ruff format --check src/ tests/ scripts/ examples/
```
Expected: clean.

- [ ] **Step 5: Run the tier-model and freeze-guard CI scripts**

```bash
.venv/bin/python scripts/cicd/enforce_tier_model.py check --root src/elspeth --allowlist config/cicd/enforce_tier_model
.venv/bin/python scripts/cicd/enforce_freeze_guards.py check
```
Expected: both green. The new files (`telemetry.py`, `_persist_payload.py`) live in L3 and should not introduce any upward imports. The new `src/elspeth/contracts/advisory_locks.py` is L0 (contracts layer); the import from L3 (`src/elspeth/web/sessions/service.py`) is downward and tier-model-clean — this is the canonical shape for a contracts-layer constant consumed by an L3 plugin (see CLAUDE.md "Layer Dependency Rules").

- [ ] **Step 6: Open the Schedule C closure PR**

```bash
gh pr create --title "test(composer): progress persistence phase 1C — PostgreSQL and CI proof" --body "$(cat <<'EOF'
## Summary

Schedule C closes Phase 1 of composer-progress-persistence after the
Schedule A schema/current-writer safety PR and Schedule B compose-turn
primitive/audit semantics PR have merged.

- Proves `initialize_session_schema(engine)` succeeds on the PostgreSQL
  schema exercised by the testcontainer lane.
- Adds or fixes the Docker-enabled `pytest -m testcontainer
  tests/integration/web/ -v` CI lane and `ci-success` gating.
- Verifies CL-PP-11, advisory-lock acquisition, cross-allocator
  serialization, stale compose-turn rejection, and latency sanity.
- Confirms dependency checks fail closed when `testcontainers.postgres`
  is unavailable.
- Amends final spec/OQ handoff text and records the Phase 1 checkpoint.

## Prior Schedule PRs

- Schedule A: schema and current writer safety PR
- Schedule B: compose-turn primitive and audit semantics PR

## Spec

`docs/superpowers/specs/2026-04-30-composer-progress-persistence-design.md` revision 4.

## Out of scope (later phases)

- Compose-loop integration (Phase 3)
- Redaction primitives (Phase 2)
- Frontend recovery panel (Phase 4)

## Follow-ups filed

- [OQ-1] elspeth-XXXXXXXX (chat_messages retention CLI extension)
- [OQ-3] to be filed in Phase 3
- [OQ-4] staging runbook documents session-DB archive/delete/restart recreation procedure (supersedes the original "pre-deploy DELETE" framing, which did not match codebase reality — no Alembic exists for the session DB)
- [Spec alignment] `docs/superpowers/specs/2026-04-30-composer-progress-persistence-design.md` amended for Phase 1 corrections: audit role, session_fork principal, no `_StatePayload.version`, `persist_compose_turn(raw_content, expected_current_state_id)`, async dispatcher on `SessionServiceProtocol`, two-field `_AuditOutcome`, `_session_write_lock`, audit_access_log cascade, production-equivalent session DB tests, and no row-level DELETE migration framing.

## Test plan

- [ ] Unit tests against in-memory SQLite (`tests/unit/web/sessions/`, `tests/unit/web/sessions/test_telemetry.py`, `tests/unit/web/composer/test_audit_failure_primacy.py`)
- [ ] Integration test: backward-direction INV-AUDIT-AHEAD (`tests/integration/web/test_inv_audit_ahead_backward.py`)
- [ ] Integration test: CL-PP-11 concurrent multi-session against testcontainer Postgres
- [ ] Integration test: per-turn latency sanity bound (p95 <= 250ms)
- [ ] tier-model CI green
- [ ] freeze-guard CI green
- [ ] mypy clean
- [ ] ruff clean

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

- [ ] **Step 7: After the PR merges, record the Phase 1 checkpoint in Filigree**

Do **not** close `elspeth-90b4542b63` after Phase 1. That feature owns
VAL for the whole user-visible recovery capability, and Phase 1 is only
the data-layer/sync-primitive slice. Use `mcp__filigree__add_comment`
(or `filigree add-comment`) on `elspeth-90b4542b63` with the Phase 1 PR
link, commit SHA, and the explicit note that phases 2-4 remain open. If
a dedicated Phase 1 child issue exists by implementation time, close
that child instead and link it from the feature comment.

(Note: do NOT close the parent epic `elspeth-528bde62bb` until all four phases are merged.)

---

## Phase 1 Done When

Schedules A, B, and C have each landed through their own review gate,
and Task 0 plus Tasks 1-20 above are complete. Specifically:

1. [ ] All new tests pass against in-memory SQLite.
2. [ ] CL-PP-11 + the B3 cross-allocator test (`test_cross_allocator_save_state_serialises_with_persist_turn`) + the B1/stale dual-`persist_compose_turn` test (`test_concurrent_persist_compose_turn_same_state_stale_rejects_without_integrity_alert`) pass against testcontainer PostgreSQL **via Task 20 Step 1b explicitly** — passing tests in the Docker-free `pytest tests/ -x` run does NOT count for this gate (the marker deselect silently skips them; B4 from the plan-review synthesis). The B1 regression closes the helper-contract-level fabricated-Tier-1-violation vector and proves stale same-state compose rejects do not increment the integrity counter.
3. [ ] The schema-level backward-direction post-condition is enforced and tested, including intra-turn ordering (B2 fix, see T15).
4. [ ] Existing `add_message` callers updated to pass `writer_principal` — including `_persist_tool_invocations`, `_persist_llm_calls`, and the `fork_session` batch-insert sweep.
5. [ ] tier-model and freeze-guard CI green.
6. [ ] Latency NFR sanity bound green (p95 <= 250 ms with N <= 8 tool calls).
7. [ ] Staging runbook documents the session-DB archive/delete/restart recreation procedure (OQ-4 superseded — see Task 18 rationale).
8. [ ] OQ-1 retention CLI follow-up Filigree ticket filed.
9. [ ] Design/spec and overview handoff text amended so downstream phases do not copy stale Phase 1 snippets.

Phase 2 begins only after the Schedule C closure PR merges. Phase 2
builds on the schema and primitive delivered here but does not require
the compose loop to be wired (Phase 3) or the frontend (Phase 4).

---

## What Phase 1 actually delivers (and does not)

**No user-visible change.** Phase 1 ships:

- New schema columns (`chat_messages.tool_call_id`, `parent_assistant_id`, `sequence_no`, `writer_principal`, `composition_states.provenance`), an internal `chat_messages.role="audit"` storage shape for unparented composer breadcrumbs, and a new `audit_access_log` table.
- A synchronous persistence primitive (`SessionServiceImpl.persist_compose_turn`) with its async-loop guard, protocol-public async dispatcher, stale-current-state guard, `_session_write_lock` + sequence-allocation helpers, and `_AuditOutcome` disposition.
- An advisory-lock classid registry (`src/elspeth/contracts/advisory_locks.py`) defining `ELSPETH_SESSIONS_LOCK_CLASSID` as on-the-wire ABI under change control — the namespace partition that makes the two-argument `pg_advisory_xact_lock` form safe in shared-cluster deployments (B3 fix). Future ELSPETH advisory locks register their classid here.
- The `_StatePayload` contract change closing B1/B3: no `version` field on the payload; `_insert_composition_state` allocates per-session `composition_states.version` under `_session_write_lock`. The dual-allocator race that previously fabricated Tier-1 audit-integrity violations (when two compose turns observed the same `state.version` in async land before the sync worker acquired the lock) is now structurally impossible. This is a contract-level fix that hardens Phase 3's compose-loop wiring against a defect class the Phase 3 plan would otherwise have inherited.
- A telemetry container (`_SessionsTelemetry`) wired into the service constructor and into `app.py`'s production wiring.
- The atomic `add_message` rewrite with the four preserved behaviours, six route/helper call-site updates, and the `fork_session` direct-insert sweep.
- Design/spec and overview handoff text amended so later phases do not copy stale Phase 1 assumptions.
- Schema-level INV-AUDIT-AHEAD backward-direction enforcement and the latency sanity bound.

The new schema columns and table have NO user-facing behaviour
because `persist_compose_turn` is not yet called by anything (the
compose loop wiring is Phase 3) and the `audit_access_log` writer
is also Phase 3 (the `include_tool_rows=true` route extension).
The redaction primitives that produce the `summarizer_errors` and
`unknown_response_key` telemetry are Phase 2.

**Why stage dormant infrastructure through three schedules.** The
schema, the sync primitive, and the audit-primacy disposition are
load-bearing for Phases 2-4, but the review found that landing them in
one PR hides too much risk. Schedule A proves the schema can coexist
with every current writer before any dormant primitive exists. Schedule
B proves the primitive's transcript, audit, and cancellation semantics
before PostgreSQL/CI infrastructure becomes the dominant concern.
Schedule C proves the same contract across PostgreSQL and CI before
Phase 1 is declared complete. This keeps later phases on a stable
persistence contract without forcing one high-blast-radius merge.

**Stall criterion (event-based).** ELSPETH does not commit to
calendar dates for Phase 2-4 (per project policy: ADR-level SLAs
are governance devices, not release deadlines). The stall
criterion for the Phase 1 dormant infrastructure is therefore
event-based:

1. **Phase 2 stalls AND ELSPETH starts onboarding real users.**
   At that point the schema-recreation runbook (Task 18) becomes
   inadequate (real users have chat history that must survive a
   schema change), and the dormant `audit_access_log` table is
   visible to every operator looking at the database without a
   writer to explain its purpose. Decision: introduce migrations
   (cancelling the no-tech-debt-pre-release exemption) OR revert
   Phase 1 to keep the schema in sync with what's actually
   wired.

2. **Phase 2 ships, Phase 3 stalls indefinitely.** The compose
   loop is the first production consumer of
   `persist_compose_turn_async`. If Phase 3 stalls, the concrete sync
   primitive remains uncalled and the async protocol dispatcher remains
   a dormant typed contract. This is acceptable only while there is no
   production caller: the sync primitive stays concrete-only, so routes
   do not accumulate direct `_run_sync` coupling, and the single public
   surface is the async dispatcher that Phase 3 will use.

3. **Phase 3 ships, Phase 4 stalls indefinitely.** Frontend is
   independent — Phase 3's audit-grade view route (with
   `include_tool_rows`) is consumable via direct API calls
   without the recovery panel UI. No revert needed.

The first criterion is the only one that forces a decision; the
second and third are operationally tolerable. Closes synthesised
review finding SA-11 / M15. The framing follows the project's
"no calendar shipping commitments" doctrine — every criterion
fires on a real-world event (user onboarding, phase merge), not a
date.
