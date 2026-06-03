# Composer Progress Persistence — Phase 1A: Schema and Current Writer Safety

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this schedule task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

> **Traceability:** Task numbers are preserved from the original Phase 1 plan so review findings can cite the same task IDs across Schedule 1A/1B/1C. Do not renumber tasks inside a schedule.

**Goal:** Land only the schema and existing-writer compatibility work needed to make the new persistence columns safe for today's code paths.

**Risk controlled:** destructive schema changes, stale staging DB bootstrap, direct writer omissions, and regressions in existing message/state persistence.

**Architecture:** This schedule is current-writer infrastructure only. It may introduce locking/sequence helpers where existing writers need them, but it must not introduce `persist_compose_turn`, `persist_compose_turn_async`, compose-loop integration, frontend work, or PostgreSQL concurrency claims beyond schema-portability smoke checks.

**Review focus:** direct table writers, schema portability, staging DB recreation, `add_message` behavior preservation, `fork_session` direct inserts, `get_messages` ordering/filtering, protocol hydration, and test-path integrity.

---

## Schedule 1A Scope

**Included tasks:** Task 0, Task 1, Task 2, Task 3, Task 4, Task 7, Task 8, Task 9, Task 10, Task 14, and Task 18.

**Explicit exclusions:** Task 5, Task 6, Task 11, Task 12, Task 13, Task 15, Task 16, Task 17, Task 19, Task 20, and all later composer/frontend phases.

**Execution order and atomicity rule:** Execute Task 0, Task 18, this direct-write inventory gate, the shared conftests, and the mechanical direct-write guard before any schema-breaking metadata lands. Tasks 1-4 may add failing tests and local metadata changes, but the schema changes that introduce NOT NULL `chat_messages.sequence_no`, NOT NULL `chat_messages.writer_principal`, and NOT NULL `composition_states.provenance` MUST NOT be committed or merged until Tasks 7-10 and Task 14 update every current production/test writer in the same atomic schema/current-writer cutover. A standalone green schema-test run is not sufficient evidence for merge because the live writers are part of the schema contract.

**Must land before Schedule 1B starts.** Schedule 1B assumes Schedule 1A's schema, writer-principal, sequence/version, provenance, and current-writer compatibility contracts are already reviewed and merged.

---

## Schedule 1A Preflight: Direct-Write Inventory Gate

Before Task 1 changes required columns, inventory every direct writer for both affected tables. This gate closes the review finding that current direct test inserts and production batch inserts can bypass `add_message` / composition-state helpers.

- [ ] **Step 1: Inventory chat-message writers**

Run:

```bash
rg -n --no-ignore-vcs "insert\((models\.)?chat_messages_table|chat_messages_table\.insert|insert\(chat_messages_table" src tests evals -g '*.py'
rg -n --no-ignore-vcs "INSERT\s+INTO\s+chat_messages|exec_driver_sql\(|raw_connection\(|cursor\.execute\(|executemany\(" src tests evals -g '*.py'
```

`--no-ignore-vcs` is mandatory. The repository's root `.gitignore` has a generic `lib/` rule (Python build artefact) that matches the directory `tests/unit/evals/lib/`; the negation entries `!tests/unit/evals/lib/` and `!tests/unit/evals/lib/**` re-include the path for git tracking but ripgrep's recursive traversal applies the parent directory exclude before the negation, so a plain `rg` invocation silently skips the tracked standalone eval-decode chat-messages writer at `tests/unit/evals/lib/test_decode_tools.py` lines 107-108 and 186. `--no-ignore-vcs` bypasses `.gitignore` while still honoring manual `.ignore` / `.rgignore` rules. Verify the eval-fixture writers appear in the output before treating the inventory as complete; if they do not, the grep is broken and the inventory is unsound.

Expected: every SQLAlchemy-table result is either rewritten in this schedule, routed through the new helper, or explicitly documented as not writing rows. Every raw-SQL result must be inspected and classified as one of: direct writer, corruption fixture that intentionally bypasses the normal writer, or unrelated raw SQL. Do not rely on the SQLAlchemy grep alone; existing tests use raw cursor SQL for integrity/corruption setup, and those sites can otherwise bypass the new required columns.

- [ ] **Step 2: Inventory composition-state writers**

Run:

```bash
rg -n --no-ignore-vcs "insert\((models\.)?composition_states_table|composition_states_table\.insert|insert\(composition_states_table" src tests evals -g '*.py'
rg -n --no-ignore-vcs "INSERT\s+INTO\s+composition_states|exec_driver_sql\(|raw_connection\(|cursor\.execute\(" src tests evals -g '*.py'
```

`--no-ignore-vcs` is mandatory for the same reason described in Step 1. The `evals` path is included alongside `src` and `tests` so any future composition-state writer added in `evals/lib/` or its tests is surfaced; the current tree has none, but extending the search anchors closes the same negation footgun for both tables.

Expected: every SQLAlchemy-table result supplies `provenance`, uses the new helper, or is rewritten to a canonical test row factory. Every raw-SQL result must be inspected and classified as one of: direct writer, corruption fixture that intentionally bypasses the normal writer, or unrelated raw SQL.

- [ ] **Step 3: Convert the inventory to a concrete edit table**

Paste a table into the PR body with these columns: table, file, line, writer kind, required 1A action, and verification. Include every production and test writer. The reviewed snapshot must include at least:

- `src/elspeth/web/sessions/service.py` — `add_message`, `save_composition_state`, `set_active_state`, and `fork_session`.
- `tests/unit/web/sessions/test_models.py` — direct chat-message rows and direct composition-state rows.
- `tests/unit/web/blobs/test_service.py` — direct composition-state rows.
- `tests/unit/web/composer/test_tools.py` — direct composition-state rows.
- `tests/unit/evals/lib/test_decode_tools.py` — standalone SQLite
  `chat_messages` fixture schema plus raw `INSERT INTO chat_messages`
  rows used by the eval decode helper tests.
- `tests/unit/web/sessions/test_fork.py` — raw FK-off corruption fixture.
- `tests/unit/web/sessions/test_routes.py` — OperationalError SQL canaries that mention
  `chat_messages` / `composition_states` without writing those tables.

Every listed direct test insert must either use the canonical row factory created in this schedule or explicitly set the new required columns with a comment naming why a direct row is still intentional. Every raw SQL / cursor / OperationalError-canary match must be classified in the inventory table even when it is a false positive; otherwise the static guard cannot distinguish intentional corruption setup from a new bypass.

- [ ] **Step 4: Add a mechanical direct-writer guard**

Before the schema/current-writer cutover commit, create `tests/unit/web/sessions/test_static_direct_writers.py`. This small static test must enumerate direct inserts to `chat_messages_table` and `composition_states_table` under `src/` and `tests/`, including SQLAlchemy `insert(...)`, table `.insert()`, raw `INSERT INTO ...` SQL, `exec_driver_sql(...)`, `raw_connection()`, and `cursor.execute(...)` sites that mention either table. It must fail when a new site appears outside the reviewed allowlist.

The guard must be mechanical, not a broad path grep. Implement it as an AST/token scanner with allowlist entries shaped like:

```python
ReviewedWriter(
    path="src/elspeth/web/sessions/service.py",
    enclosing_symbol="SessionServiceImpl.add_message._sync",
    table="chat_messages",
    operation="sqlalchemy_insert",
    purpose="current add_message writer updated by Task 14",
)
```

For raw strings / DB-API cursor operations, key the allowlist by `path`, `enclosing_symbol`, matched table name, operation kind, and semantic purpose (`standalone_eval_fixture`, `corruption_fixture`, `operational_error_canary`, `unrelated_raw_sql`, etc.). This includes raw `INSERT INTO ...` strings passed through `sqlite3` / DB-API `execute(...)` or `executemany(...)`, not only SQLAlchemy `exec_driver_sql(...)` or explicit `cursor.execute(...)` calls. A new direct writer inside an already-allowed file must still fail unless it is inside the same reviewed semantic site. The scanner must ignore its own scanner patterns in `test_static_direct_writers.py` so the test does not allowlist itself by accident.

The same static guard must also check lock-sensitive helper usage: calls to `_reserve_sequence_range(...)`, `_insert_chat_message(...)`, and `_insert_composition_state(...)` must either be inside `with self._session_write_lock(conn, session_id):` in the same enclosing symbol or be explicitly allowlisted as a negative test. The helpers themselves must also call `_assert_session_write_lock_held(...)`, so the precondition is enforced at runtime and by static drift checks. For inline `composition_states.version` allocation, the guard must reject a `SELECT MAX(composition_states.version)` / insert pair in `save_composition_state` or `set_active_state` unless the enclosing AST block is inside `_session_write_lock`. This closes the review finding that lock discipline cannot remain comment-only.

The same cutover must add canonical test row factories for any remaining intentional direct inserts, so future tests do not silently omit `sequence_no`, `writer_principal`, or `provenance`.

Required guard tests inside `tests/unit/web/sessions/test_static_direct_writers.py`:

1. `test_static_direct_writers_match_reviewed_allowlist` — scans the live `src/` and `tests/` tree and fails on any unreviewed SQLAlchemy/table/raw writer match.
2. `test_static_direct_writer_guard_rejects_unreviewed_chat_insert` — feeds the scanner a synthetic unallowlisted `chat_messages_table.insert()` site and asserts it fails closed.
3. `test_static_direct_writer_guard_rejects_unreviewed_state_insert` — feeds the scanner a synthetic unallowlisted `composition_states_table.insert()` site and asserts it fails closed.
4. `test_static_helper_lock_guard_rejects_unlocked_allocator` — feeds the scanner a synthetic `_reserve_sequence_range(...)` / `_insert_chat_message(...)` / `_insert_composition_state(...)` call outside `_session_write_lock` and asserts it fails closed.

Run the static guard explicitly:

```bash
.venv/bin/python -m pytest tests/unit/web/sessions/test_static_direct_writers.py -v
```

Expected before the guard implementation exists: FAIL. Expected after the scanner, reviewed allowlist, and synthetic negative fixtures land: PASS. Do not count the guard as review evidence unless the synthetic negative tests prove it rejects a new bypass without modifying production source.

- [ ] **Step 5: Commit the inventory result before the schema/current-writer cutover**

Record the inventory output summary in the PR body. The schema PR is not reviewable without it.

---

## Task 0: Mark stale spec snippets superseded before implementation

**Why this is first.** Multiple Phase 1 corrections intentionally
supersede stale text in
`docs/superpowers/specs/2026-04-30-composer-progress-persistence-design.md`.
Leaving that spec as the governing handoff until Task 19 invites the
next implementer to copy the wrong lock SQL, stale payload
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
- `_insert_composition_state` accepts `CompositionStateData` directly
  and allocates versions under `_session_write_lock`; no 1A caller
  supplies a precomputed state version.
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
- Create: `tests/unit/web/conftest.py` (test helpers — applies to every Phase 1 unit test under `tests/unit/web/...`, including both `tests/unit/web/sessions/` and `tests/unit/web/composer/`. Hoisted to the parent package so the `_make_session` helper and `engine` fixture are visible to composer-suite tests without duplication. Synthesised review B5.)
- Create: `tests/integration/web/conftest.py` (test helpers — duplicates the `_make_session` helper for integration tests; the helper is one ~15-line function and parallel conftests is the standard pytest pattern for this codebase, see `tests/integration/checkpoint/conftest.py`)
- Modify: `src/elspeth/web/sessions/models.py`
- Test: `tests/unit/web/sessions/test_chat_messages.py` (create)

**Fixture reach note.** A parent `tests/unit/web/conftest.py` makes the shared helpers available to tests under `tests/unit/web/...`, but it does not override local fixtures that already define `engine` in files such as `tests/unit/web/sessions/test_service.py`, `tests/unit/web/sessions/test_fork.py`, and `tests/unit/web/sessions/test_models.py`. This schedule may migrate those local fixtures when touched, but the parent conftest must be described as a shared helper source for new/updated tests, not as proof that every existing web test automatically uses the new fixture.

- [ ] **Step 1a: Create the unit-test conftest**

Create `tests/unit/web/conftest.py`:

```python
"""Shared fixtures and helpers for Phase 1 web unit tests.

Hoisted from ``tests/unit/web/sessions/conftest.py`` to the parent
``tests/unit/web/`` package so both the sessions suite
(``tests/unit/web/sessions/test_*.py``) and the composer suite see the
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
integration tests own their engine fixture and call ``_make_session``
against whatever connection they have.
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

- [ ] **Step 1c: Commit the shared conftests before any later task imports them**

The shared `_make_session` helpers do not depend on the rev-4 schema and
must be available from a clean checkout before Task 7 creates
`tests/unit/web/sessions/test_persist_compose_turn.py`. Commit them as a
standalone test-support change now; otherwise the Task 7 commit imports a
file that is still only present in an uncommitted worktree.

```bash
git add tests/unit/web/conftest.py tests/integration/web/conftest.py
git commit -m "test(web): add shared session row factories for phase 1 persistence tests"
```

- [ ] **Step 1d: Write the failing schema test**

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
from sqlalchemy import delete, insert, select
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

This schema proves the parent message exists in the same session. It does **not** prove the parent row has `role='assistant'`; portable SQLAlchemy CHECK constraints cannot inspect a referenced row. Task 9 therefore adds `_assert_parent_assistant_message(...)` and requires every service writer that can create `role='tool'` rows to call through `_insert_chat_message` or perform the same guard before direct batch insert.

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

**Constraint-isolation gate after metadata lands.** The initial red run
may fail because required columns are absent before it reaches the
intended CHECK/FK/unique predicates. After Step 3 lands metadata, rerun
the named tests for each invariant and confirm the failure/pass signal
is the intended contract, not a missing-column accident:

```bash
.venv/bin/python -m pytest tests/unit/web/sessions/test_chat_messages.py -v -k "tool_call_id_role or parent_role or writer_principal or unique or cascades or cross_session_parent"
```

The PR body must call out any test whose original red was coarse and
confirm the post-metadata rerun isolated the intended CHECK, FK, or
unique-index behaviour.

- [ ] **Step 5: Do not commit the schema metadata standalone**

Stop here with the tests and metadata changes verified locally. Per the 1A atomicity rule, these `chat_messages` schema changes are committed only in Task 14's schema/current-writer cutover, after `add_message`, `fork_session`, and all direct test writers supply `sequence_no` and `writer_principal`. Task 14's commit command stages `src/elspeth/web/sessions/models.py` and `tests/unit/web/sessions/test_chat_messages.py`; the shared conftests were already committed in Step 1c so Task 7 and later clean checkouts can import `_make_session`.

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


def test_tool_call_id_may_repeat_across_sessions(engine):
    """The unique index scope is (session_id, tool_call_id), not
    tool_call_id globally. Two sessions may receive the same provider
    tool_call_id without colliding."""
    with engine.begin() as conn:
        _make_session(conn, session_id="s1")
        _make_session(conn, session_id="s2")
        for sid, assistant_id, tool_id, seq in (
            ("s1", "a1", "t1", 1),
            ("s2", "a2", "t2", 1),
        ):
            conn.execute(insert(models.chat_messages_table).values(
                id=assistant_id, session_id=sid, role="assistant",
                content="", sequence_no=seq, writer_principal="compose_loop",
                created_at=datetime(2026, 4, 30, tzinfo=UTC),
            ))
            conn.execute(insert(models.chat_messages_table).values(
                id=tool_id, session_id=sid, role="tool",
                content="{}", sequence_no=seq + 1, writer_principal="compose_loop",
                tool_call_id="same_provider_id",
                parent_assistant_id=assistant_id,
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
.venv/bin/python -m pytest tests/unit/web/sessions/test_chat_messages.py -v -k "tool_call_id_unique_within_session or tool_call_id_may_repeat_across_sessions"
```
Expected: `test_tool_call_id_unique_within_session` FAILS because the partial unique index does not exist. `test_tool_call_id_may_repeat_across_sessions` may already pass before the index exists; it is still part of the green gate because it proves the index was not accidentally scoped to `tool_call_id` alone.

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
Expected: PASS for the three new tests; previously-passing tests still pass. The cross-session duplicate test is required evidence that the index is scoped by `session_id`.

- [ ] **Step 5: Do not commit the schema metadata standalone**

Keep this partial-index change with Task 1's `chat_messages` metadata and Task 14's current-writer updates. The partial index is harmless by itself, but splitting it into its own commit invites a false sense that the schema layer is independently mergeable; 1A treats the full chat-message schema and writer cutover as one review boundary.

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

**Service-record hydration decision.** In Schedule 1A, `provenance` is a
DB-enforced audit column, not a public API field. Do not add it to
`CompositionStateRecord` or `CompositionStateResponse` in this schedule
unless a current 1A caller proves it needs the value. The 1A proof is the
write-side invariant: every current writer supplies a truthful
provenance enum and the database rejects omissions or unknown values.
If Schedule 1B or a later audit-view route needs read-side provenance,
that schedule must add explicit protocol/response hydration and tests
instead of assuming this DB-only 1A column is already surfaced.

- [ ] **Step 4: Run tests to verify they pass**

```bash
.venv/bin/python -m pytest tests/unit/web/sessions/test_composition_states.py -v
```
Expected: PASS.

- [ ] **Step 5: Do not commit the schema metadata standalone**

Stop here with the tests and metadata verified locally. Per the 1A atomicity rule, `composition_states.provenance` is committed only in Task 10's state-writer cutover, after `save_composition_state`, `set_active_state`, `fork_session`, and all direct test inserts provide provenance.

---
## Task 4: Add `audit_access_log` table

**Why this future-facing table is still in 1A.** No 1A route writes
`audit_access_log`; the audit-grade message-view route remains a later
phase. The table lands here because 1A is already the destructive
session-DB schema reset boundary. Deferring this table to a later phase
would force a second staging DB recreation for a table whose ownership,
FK shape, and writer-principal enum are already known. Treat the table as
inert schema in 1A: do not add public `include_tool_rows` / audit-view
route behaviour in this schedule, and do not count table existence as
proof that audit-access logging is implemented.

**Future-writer privacy gate.** The table contains privacy-sensitive
request context (`requesting_principal`, `request_path`, `query_args`,
and `ip_address`). Before any later schedule adds a writer, that schedule
must define and test an allowlist for query arguments, must prove it never
stores headers, request bodies, secrets, provider tokens, or arbitrary
exception strings, and must choose an explicit IP retention policy
(literal storage, truncation, or keyed hash). 1A lands only the inert
table and the writer-principal CHECK; it does not authorize a writer.

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

- [ ] **Step 5: Do not commit schema metadata separately from the 1A cutover**

Keep the new table with the same schema/current-writer cutover review boundary. Task 14's final atomic staging list includes `src/elspeth/web/sessions/models.py` and all schema tests created in Tasks 1-4.

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

**Deploy preflight for SQLite process-local locking.** Before deploying
1A against a SQLite session DB, verify that the live service is still
single-worker: `deploy/elspeth-web.service` must not pass `--workers`,
`WEB_CONCURRENCY` in `deploy/elspeth-web.env` must be unset or `1`, and
the startup guard in `src/elspeth/web/app.py` must remain enabled. If any
deployment target uses multiple workers or multiple processes against
the same SQLite file, stop: 1A's SQLite branch is not safe for that
target without a cross-process locking design. PostgreSQL session DB
deployments must wait for Schedule 1C's PostgreSQL proof before relying
on the Postgres branch operationally.

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
    return SessionServiceImpl(engine, data_dir=tmp_path)


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

In `src/elspeth/web/sessions/service.py`, add the helpers inside the `SessionServiceImpl` class. Note the import of `ELSPETH_SESSIONS_LOCK_CLASSID` at module top — cite the constant by name at every call site so future grep finds them all (open-coded literals defeat the registry pattern). Add `import contextlib`, `import threading`, and `from contextvars import ContextVar`, and define the SQLite lock registry plus lock-held token at module top so every service instance in the process serializes on the same locks:

```python
_SQLITE_SESSION_LOCKS_GUARD = threading.RLock()
_SQLITE_SESSION_LOCKS: dict[tuple[str, str], threading.RLock] = {}
_SESSION_WRITE_LOCK_HELD: ContextVar[frozenset[tuple[int, str]]] = ContextVar(
    "_SESSION_WRITE_LOCK_HELD",
    default=frozenset(),
)
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


def _assert_session_write_lock_held(
    self,
    conn: Connection,
    session_id: str,
    *,
    caller: str,
) -> None:
    """Mechanical precondition guard for session-scoped allocators.

    The docstring precondition on _reserve_sequence_range and
    _insert_composition_state is not enough: future callers can forget the
    lock and still pass type checks. _session_write_lock sets a per-thread
    ContextVar token keyed by (id(conn), session_id); lock-requiring helpers
    crash immediately if called without that token in the same transaction.
    """
    if (id(conn), session_id) not in _SESSION_WRITE_LOCK_HELD.get():
        raise RuntimeError(
            f"{caller}: _session_write_lock(conn, {session_id!r}) must be "
            "held in the same transaction before allocating session-scoped "
            "sequence/version values"
        )


@contextlib.contextmanager
def _session_write_lock(self, conn: Connection, session_id: str):
    """Serialize same-session sequence/version allocators.

    PostgreSQL uses the transaction-scoped advisory lock. SQLite uses a
    process-wide per-session RLock around the whole allocator + insert
    sequence. Every caller that performs ``SELECT MAX(...) + 1`` for
    ``chat_messages.sequence_no`` or ``composition_states.version`` MUST
    wrap that read and every dependent INSERT in this context.
    """
    key = (id(conn), session_id)
    held = _SESSION_WRITE_LOCK_HELD.get()
    token = _SESSION_WRITE_LOCK_HELD.set(held | {key})
    dialect = self._engine.dialect.name
    try:
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
    finally:
        _SESSION_WRITE_LOCK_HELD.reset(token)
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
        with service._session_write_lock(conn, "s1"):
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
        with service._session_write_lock(conn, "s2"):
            base = service._reserve_sequence_range(conn, "s2", count=2)
        assert base == 6


def test_reserve_sequence_range_requires_session_write_lock(service):
    with service._engine.begin() as conn:
        _make_session(conn, session_id="s_no_lock")
        with pytest.raises(RuntimeError, match="_session_write_lock"):
            service._reserve_sequence_range(conn, "s_no_lock", count=1)


@pytest.mark.timeout(5)
def test_session_write_lock_serializes_sqlite_same_session_sequence_allocation(service):
    """B3 regression: two same-session SQLite writers must not both read
    the same MAX(sequence_no). This test uses the real StaticPool
    in-memory SQLite engine from the shared fixture and two worker
    threads. The sleep happens inside the session write lock to widen
    the race window; without the process-wide per-session lock both
    workers can reserve sequence_no=1 and one insert fails."""
    from concurrent.futures import ThreadPoolExecutor, wait
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

    pool = ThreadPoolExecutor(max_workers=2)
    try:
        futures = [pool.submit(_writer, index) for index in (1, 2)]
        done, not_done = wait(futures, timeout=2.0)
        assert not not_done, (
            "SQLite same-session sequence-allocation workers did not finish "
            "within 2s; likely deadlock or lock-order regression"
        )
        seqs = sorted(future.result(timeout=0) for future in done)
    finally:
        pool.shutdown(wait=False, cancel_futures=True)

    assert seqs == [1, 2]
    with service._engine.begin() as conn:
        persisted = conn.execute(
            select(models.chat_messages_table.c.sequence_no)
            .where(models.chat_messages_table.c.session_id == "s_sqlite_lock")
            .order_by(models.chat_messages_table.c.sequence_no)
        ).scalars().all()
    assert persisted == [1, 2]


def test_file_backed_sqlite_sequence_allocator_smoke(tmp_path):
    """Staging uses file-backed SQLite, not only StaticPool in-memory SQLite.
    This smoke proves the same helper path works against a temporary file DB."""
    from elspeth.web.sessions.engine import create_session_engine
    from elspeth.web.sessions.schema import initialize_session_schema

    db_path = tmp_path / "sessions.db"
    engine = create_session_engine(f"sqlite:///{db_path}")
    initialize_session_schema(engine)
    service = SessionServiceImpl(engine, data_dir=tmp_path)
    with service._engine.begin() as conn:
        _make_session(conn, session_id="s_file")
        with service._session_write_lock(conn, "s_file"):
            assert service._reserve_sequence_range(conn, "s_file", count=1) == 1


@pytest.mark.timeout(5)
def test_file_backed_sqlite_lock_serializes_independent_connections(tmp_path):
    """Staging uses file-backed SQLite with independently checked-out
    connections. This is the representative race proof; the StaticPool
    in-memory test above is only fixture/reentrancy coverage."""
    from concurrent.futures import ThreadPoolExecutor, wait
    import threading
    import time

    from elspeth.web.sessions import models
    from elspeth.web.sessions.engine import create_session_engine
    from elspeth.web.sessions.schema import initialize_session_schema
    from sqlalchemy import insert, select

    db_path = tmp_path / "sessions.db"
    engine = create_session_engine(f"sqlite:///{db_path}")
    initialize_session_schema(engine)
    service = SessionServiceImpl(engine, data_dir=tmp_path)
    with engine.begin() as conn:
        _make_session(conn, session_id="s_file_concurrent")

    barrier = threading.Barrier(2)

    def _writer(index: int) -> int:
        barrier.wait()
        with engine.begin() as conn:
            with service._session_write_lock(conn, "s_file_concurrent"):
                seq = service._reserve_sequence_range(
                    conn, "s_file_concurrent", count=1
                )
                time.sleep(0.01)
                conn.execute(insert(models.chat_messages_table).values(
                    id=f"m_file_{index}",
                    session_id="s_file_concurrent",
                    role="user",
                    content=f"message {index}",
                    sequence_no=seq,
                    writer_principal="route_user_message",
                    created_at=datetime(2026, 4, 30, tzinfo=UTC),
                ))
                return seq

    pool = ThreadPoolExecutor(max_workers=2)
    try:
        futures = [pool.submit(_writer, index) for index in (1, 2)]
        done, not_done = wait(futures, timeout=2.0)
        assert not not_done, (
            "File-backed SQLite sequence-allocation workers did not finish "
            "within 2s; likely deadlock or lock-order regression"
        )
        seqs = sorted(future.result(timeout=0) for future in done)
    finally:
        pool.shutdown(wait=False, cancel_futures=True)

    assert seqs == [1, 2]
    with engine.begin() as conn:
        persisted = conn.execute(
            select(models.chat_messages_table.c.sequence_no)
            .where(models.chat_messages_table.c.session_id == "s_file_concurrent")
            .order_by(models.chat_messages_table.c.sequence_no)
        ).scalars().all()
    assert persisted == [1, 2]
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv/bin/python -m pytest tests/unit/web/sessions/test_persist_compose_turn.py -v -k "reserve_sequence_range or session_write_lock_serializes_sqlite_same_session_sequence_allocation or file_backed_sqlite_sequence_allocator_smoke or file_backed_sqlite_lock_serializes_independent_connections"
```
Expected: FAIL — helper does not exist. The command must include the same-session and file-backed SQLite race tests; `-k reserve_sequence` alone does not match them and is not a valid lock-safety red run.

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
    callers to invoke them in order. Schedule 1A has the current
    add-message path and ``fork_session`` copy path; Schedule 1B adds
    ``persist_compose_turn``. Every caller must enter
    ``_session_write_lock`` before reserving a sequence range. If a
    later phase adds another invocation
    site, consider consolidating into a single
    ``_write_chat_messages_atomic(conn, session_id, rows)`` entry
    point that hides the protocol. Until that third site appears,
    the consolidation is not justified.
    """
    self._assert_session_write_lock_held(
        conn,
        session_id,
        caller="_reserve_sequence_range",
    )
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
.venv/bin/python -m pytest tests/unit/web/sessions/test_persist_compose_turn.py -v -k "reserve_sequence_range or session_write_lock_serializes_sqlite_same_session_sequence_allocation or file_backed_sqlite_sequence_allocator_smoke or file_backed_sqlite_lock_serializes_independent_connections"
```
Expected: PASS for the helper tests and both SQLite race proofs. A pass that omits either concurrency test is a false green.

- [ ] **Step 5: Do not commit this helper standalone**

Keep the helper and tests with the schema/current-writer cutover. `_reserve_sequence_range` depends on `chat_messages.sequence_no`, so a standalone helper commit before the schema metadata is not independently verifiable.

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
        with service._session_write_lock(conn, "s3"):
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
        with service._session_write_lock(conn, "s3_raw"):
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


def test_insert_chat_message_requires_session_write_lock(service):
    """The helper is the actual chat-row writer, so the lock precondition
    must be mechanical instead of docstring-only. A future caller that
    skips `_reserve_sequence_range` must still crash before writing an
    arbitrary caller-supplied sequence number."""
    from datetime import UTC, datetime

    with service._engine.begin() as conn:
        _make_session(conn, session_id="s3_no_lock")
        with pytest.raises(RuntimeError, match="_session_write_lock"):
            service._insert_chat_message(
                conn,
                session_id="s3_no_lock",
                role="assistant",
                content="hello",
                raw_content=None,
                tool_calls=None,
                sequence_no=1,
                writer_principal="compose_loop",
                composition_state_id=None,
                tool_call_id=None,
                parent_assistant_id=None,
                created_at=datetime.now(UTC),
            )


def test_insert_chat_message_rejects_tool_parent_that_is_not_assistant(service):
    """The DB FK proves same-session parent existence only. The service
    writer must mechanically enforce that tool parents are assistant rows."""
    from datetime import UTC, datetime
    from elspeth.web.sessions import models
    from sqlalchemy import insert

    now = datetime.now(UTC)
    with service._engine.begin() as conn:
        _make_session(conn, session_id="s3_parent_role")
        conn.execute(insert(models.chat_messages_table).values(
            id="u_parent",
            session_id="s3_parent_role",
            role="user",
            content="not an assistant",
            sequence_no=1,
            writer_principal="route_user_message",
            created_at=now,
        ))
        with service._session_write_lock(conn, "s3_parent_role"):
            with pytest.raises(RuntimeError, match="parent_assistant_id.*assistant"):
                service._insert_chat_message(
                    conn,
                    session_id="s3_parent_role",
                    role="tool",
                    content="{}",
                    raw_content=None,
                    tool_calls=None,
                    sequence_no=2,
                    writer_principal="compose_loop",
                    composition_state_id=None,
                    tool_call_id="tc_1",
                    parent_assistant_id="u_parent",
                    created_at=now,
                )
```

- [ ] **Step 2: Run test to verify it fails**

```bash
.venv/bin/python -m pytest tests/unit/web/sessions/test_persist_compose_turn.py -v -k insert_chat_message
```
Expected: FAIL — `_insert_chat_message` and its lock/parent/raw-content safeguards do not exist yet. The command must run every `insert_chat_message` test in this task, not just the happy path.

- [ ] **Step 3: Implement the helper**

In `src/elspeth/web/sessions/service.py`:

```python
def _assert_parent_assistant_message(
    conn: Connection,
    *,
    parent_assistant_id: str,
    session_id: str,
    caller: str,
) -> None:
    """Offensive guard for tool rows.

    The composite FK on `(parent_assistant_id, session_id)` can prove
    same-session existence, but SQL CHECK constraints cannot portably
    inspect the referenced row's role. Service writers must therefore
    reject tool rows whose parent id exists but is not an assistant.
    """
    role = conn.execute(
        select(models.chat_messages_table.c.role).where(
            models.chat_messages_table.c.id == parent_assistant_id,
            models.chat_messages_table.c.session_id == session_id,
        )
    ).scalar_one_or_none()
    if role != "assistant":
        raise RuntimeError(
            f"{caller}: parent_assistant_id must reference an assistant "
            f"message in the same session; got role={role!r}"
        )


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

    If `role == "tool"`, this helper also verifies that
    `parent_assistant_id` references an assistant row in the same
    session. The DB FK only proves same-session existence, not parent
    role.
    """
    self._assert_session_write_lock_held(
        conn,
        session_id,
        caller="_insert_chat_message",
    )
    if role == "tool":
        if parent_assistant_id is None:
            raise RuntimeError("_insert_chat_message: tool row requires parent_assistant_id")
        _assert_parent_assistant_message(
            conn,
            parent_assistant_id=parent_assistant_id,
            session_id=session_id,
            caller="_insert_chat_message",
        )
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

(Add `import uuid` and `from datetime import datetime` if not already imported, and confirm `from sqlalchemy import insert, select` and `from . import models` are imported.)

- [ ] **Step 4: Run test to verify it passes**

```bash
.venv/bin/python -m pytest tests/unit/web/sessions/test_persist_compose_turn.py -v -k insert_chat_message
```
Expected: PASS for the happy path, `raw_content` persistence, lock-precondition, and non-assistant-parent rejection tests. A pass that omits the negative tests is a false green.

- [ ] **Step 5: Do not commit this helper standalone**

Keep the helper and tests with the schema/current-writer cutover. `_insert_chat_message` writes the new `sequence_no`, `writer_principal`, `tool_call_id`, and `parent_assistant_id` fields and must land atomically with the metadata and call-site sweep.

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
  see prologue below). The helper accepts `CompositionStateData`
  plus `derived_from_state_id`; callers cannot pass a precomputed
  version, so the dual-allocator race is closed structurally.
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
*signature* still let callers supply a precomputed ``version``. In the
Phase 3 compose loop, that version is
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

The fix is to make `_insert_composition_state` accept
`CompositionStateData` directly and have the helper allocate the
version inside the same transaction that holds the lock:

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
- Modify/test as inventory requires: `tests/unit/web/sessions/test_models.py`, `tests/unit/web/blobs/test_service.py`, and `tests/unit/web/composer/test_tools.py` so every direct `composition_states_table` insert either uses the canonical row factory or explicitly supplies `provenance`.

- [ ] **Step 1: Write the failing test**

```python
def test_insert_composition_state_returns_id(service):
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
                # B1: no ``version=``. The helper allocates it.
                state=CompositionStateData(
                    source={"kind": "tool_response"},
                    nodes=[],
                    edges=[],
                    outputs=[],
                    metadata_={},
                    is_valid=True,
                    validation_errors=None,
                ),
                derived_from_state_id=None,
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
        assert rows[0].derived_from_state_id is None


def test_insert_composition_state_allocates_contiguous_versions(service):
    """B1 (Phase 1 plan-review synthesis): under the held advisory
    lock, repeated calls to ``_insert_composition_state`` for the same
    session allocate contiguous versions starting at 1. The test runs
    serially within a single transaction. The concurrent state-version
    allocator proof for current 1A helpers is the SQLite race regression
    immediately below; Task 16 owns only the later
    ``persist_compose_turn`` stale-state intent check."""
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
                        state=CompositionStateData(),
                        derived_from_state_id=None,
                        provenance="session_seed",
                    )
                )
        rows = conn.execute(text(
            "SELECT id, version FROM composition_states "
            "WHERE session_id='s4_seq' ORDER BY version"
        )).fetchall()
    assert [r.version for r in rows] == [1, 2, 3]
    assert [r.id for r in rows] == ids


@pytest.mark.timeout(5)
def test_session_write_lock_serializes_sqlite_same_session_state_version_allocation(service):
    """Current 1A state writers also use SELECT MAX(version)+1.
    Two same-session SQLite writers must not both reserve version 1."""
    from concurrent.futures import ThreadPoolExecutor, wait
    import threading
    import time

    from elspeth.web.sessions import models
    from elspeth.web.sessions.protocol import CompositionStateData
    from sqlalchemy import select

    barrier = threading.Barrier(2)
    with service._engine.begin() as conn:
        _make_session(conn, session_id="s4_state_lock")

    def _writer(index: int) -> int:
        barrier.wait()
        with service._engine.begin() as conn:
            with service._session_write_lock(conn, "s4_state_lock"):
                state_id = service._insert_composition_state(
                    conn,
                    session_id="s4_state_lock",
                    state=CompositionStateData(metadata_={"index": index}),
                    derived_from_state_id=None,
                    provenance="session_seed",
                )
                time.sleep(0.01)
                row = conn.execute(
                    select(models.composition_states_table.c.version).where(
                        models.composition_states_table.c.id == state_id
                    )
                ).scalar_one()
                return int(row)

    pool = ThreadPoolExecutor(max_workers=2)
    try:
        futures = [pool.submit(_writer, index) for index in (1, 2)]
        done, not_done = wait(futures, timeout=2.0)
        assert not not_done, (
            "SQLite same-session state-version workers did not finish "
            "within 2s; likely deadlock or lock-order regression"
        )
        versions = sorted(future.result(timeout=0) for future in done)
    finally:
        pool.shutdown(wait=False, cancel_futures=True)

    assert versions == [1, 2]


def test_insert_composition_state_versions_are_per_session(service):
    """B1 (Phase 1 plan-review synthesis): the
    ``SELECT COALESCE(MAX(version), 0) + 1`` allocation MUST filter by
    ``session_id``. ``uq_composition_state_version`` is a per-session
    constraint (see Task 1's CREATE TABLE); a global MAX would produce
    versions that are unique cluster-wide but not contiguous within a
    session, breaking the per-session monotonic-version contract every
    other read path assumes (cf. Task 11's
    ``assert states[0].version == 1`` after a fresh session)."""
    from elspeth.web.sessions.protocol import CompositionStateData

    with service._engine.begin() as conn:
        _make_session(conn, session_id="s_ver_a")
        _make_session(conn, session_id="s_ver_b")
        with service._session_write_lock(conn, "s_ver_a"):
            for _ in range(5):
                service._insert_composition_state(
                    conn,
                    session_id="s_ver_a",
                    state=CompositionStateData(),
                    derived_from_state_id=None,
                    provenance="session_seed",
                )

    # New transaction; new session. Allocation should restart at 1
    # (per-session), NOT continue at 6 (global).
    with service._engine.begin() as conn:
        with service._session_write_lock(conn, "s_ver_b"):
            service._insert_composition_state(
                conn,
                session_id="s_ver_b",
                state=CompositionStateData(),
                derived_from_state_id=None,
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


def test_insert_composition_state_requires_session_write_lock(service):
    from elspeth.web.sessions.protocol import CompositionStateData

    with service._engine.begin() as conn:
        _make_session(conn, session_id="s4_no_lock")
        with pytest.raises(RuntimeError, match="_session_write_lock"):
            service._insert_composition_state(
                conn,
                session_id="s4_no_lock",
                state=CompositionStateData(),
                derived_from_state_id=None,
                provenance="session_seed",
            )


def test_insert_composition_state_rejects_unknown_provenance(service):
    from sqlalchemy.exc import IntegrityError
    from elspeth.web.sessions.protocol import CompositionStateData

    with service._engine.begin() as conn:
        _make_session(conn, session_id="s5")
        # Precondition contract: session write-lock first (see B1/B3 test above).
        with service._session_write_lock(conn, "s5"):
            with pytest.raises(IntegrityError, match="ck_composition_states_provenance"):
                service._insert_composition_state(
                    conn,
                    session_id="s5",
                    state=CompositionStateData(),
                    derived_from_state_id=None,
                    provenance="rogue_value",
                ),
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv/bin/python -m pytest tests/unit/web/sessions/test_persist_compose_turn.py -v -k "insert_composition_state or session_write_lock_serializes_sqlite_same_session_state_version_allocation"
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
    state: CompositionStateData,
    derived_from_state_id: str | None,
    provenance: str,
    created_at: datetime | None = None,
    state_id: str | None = None,
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

    The optional ``state_id`` exists for `fork_session`, which already
    precomputes `copied_state_id_str` and uses that same id for chat-row
    `composition_state_id` FKs and returned records. Other callers leave
    it `None` and let the helper allocate a fresh UUID.
    """
    self._assert_session_write_lock_held(
        conn,
        session_id,
        caller="_insert_composition_state",
    )
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
    state_id = state_id or str(uuid.uuid4())
    conn.execute(
        insert(models.composition_states_table).values(
            id=state_id,
            session_id=session_id,
            version=int(next_version),
            source=_enveloped_state_column(state.source),
            nodes=_enveloped_state_column(state.nodes),
            edges=_enveloped_state_column(state.edges),
            outputs=_enveloped_state_column(state.outputs),
            metadata_=_enveloped_state_column(state.metadata_),
            is_valid=state.is_valid,
            validation_errors=deep_thaw(state.validation_errors),
            derived_from_state_id=derived_from_state_id,
            provenance=provenance,
            created_at=created_at if created_at is not None else datetime.now(UTC),
        )
    )
    return state_id
```

(Add `Any` to the typing imports, and add `from elspeth.contracts.freeze import deep_thaw` and `from elspeth.web.sessions.protocol import CompositionStateData` to the imports if not already present. ``select`` and ``func`` are already imported at the module top — they are used by ``save_composition_state`` and ``_reserve_sequence_range`` for the same SQLAlchemy 2.x SELECT-MAX idiom; no new import is required for the B1 version-allocation query. Replace the existing method-local `_enveloped` helpers in `save_composition_state` and `fork_session` with `_enveloped_state_column` in this same task.)

Then make the per-site updates. There are **three** existing inline
`composition_states` inserts in `service.py`; locate them via
`grep -n 'insert(composition_states_table\|insert(models.composition_states_table)' src/elspeth/web/sessions/service.py`
(line numbers may have drifted from the snapshot below).

**Site-by-site mapping:**

| Site | Method | Treatment | Provenance |
|---|---|---|---|
| `service.py:~403` | `save_composition_state` (general route-level state save) | KEEP inline. Three additions: (1) add `provenance="session_seed",` to the existing `.values(...)` call; (2) wrap the existing `SELECT MAX(version)` + INSERT retry body in `with self._session_write_lock(conn, sid):` as the FIRST operation inside the `engine.begin()` block at `_try_insert_state` (line ~396); (3) replace the local `_enveloped` helper with shared `_enveloped_state_column`. The context makes the SELECT-then-INSERT sequence atomic against every other writer for this `session_id` — closing B3 from the Phase 1 plan-review synthesis. The retry loop is kept as belt-and-suspenders (see B3 prologue above). | `"session_seed"` (broadened semantics — spec §4.1.2 amendment in Task 3) |
| `service.py:~834` | `set_active_state` (same-session state revert/pin) | KEEP inline. Two additions: (1) add `provenance="session_seed",` to the `.values(...)` call; (2) wrap the prior-row SELECT, `SELECT MAX(version)`, and INSERT in `with self._session_write_lock(conn, sid):` inside the `engine.begin()` block at `_try_insert_revert` (line ~794). The `derived_from_state_id` is already populated correctly. The retry loop is kept as belt-and-suspenders (see B3 prologue above). | `"session_seed"` (same broadened semantics) |
| `service.py:~1191` | `fork_session` (cross-session state copy at fork) | REFACTOR to call `_insert_composition_state`. Build a `CompositionStateData` from the source state's fields and call the helper with `provenance="session_fork"`, `created_at=now`, and `state_id=copied_state_id_str`. Note: §14.6's fork sweep already requires entering `_session_write_lock` for the new session_id BEFORE this state insert (see §14.6 Step 1.5). The fork is a single-shot insert with no retry loop, which matches the helper's contract. | `"session_fork"` (new enum value — see Task 3) |

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
        # B1 (Phase 1 plan-review synthesis): no ``version=``. The
        # helper allocates COALESCE(MAX(version),0)+1 = 1 (this is the
        # first state in a freshly created session) under the held
        # session write lock. Removing the kwarg makes the
        # dual-allocator race structurally impossible; previously a
        # caller could supply a stale version computed outside the lock.
        state=CompositionStateData(
            source=source_state_record.source,
            nodes=source_state_record.nodes,
            edges=source_state_record.edges,
            outputs=source_state_record.outputs,
            metadata_=source_state_record.metadata_,
            is_valid=source_state_record.is_valid,
            validation_errors=source_state_record.validation_errors,
        ),
        derived_from_state_id=None,
        provenance="session_fork",
        created_at=now,  # cross-table timestamp consistency — pre-computed at line ~1104
        state_id=copied_state_id_str,
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

**Public-writer locking proof.** Because the existing retry loops in
`save_composition_state` and `set_active_state` could otherwise mask a
missing lock, extend `tests/unit/web/sessions/test_static_direct_writers.py`
in this task to assert both public methods enter `_session_write_lock`
before their `SELECT MAX(composition_states.version)` allocation and
before the dependent insert. If the implementation shape makes that AST
assertion brittle, add a bounded public-method concurrency regression
instead: concurrently call `save_composition_state(...)` and
`set_active_state(...)` against the same file-backed SQLite session and
assert contiguous versions without `IntegrityError`. Do not count the
helper-only `_insert_composition_state` tests as proof that these public
writers are locked.

- [ ] **Step 4: Run tests to verify pass**

```bash
.venv/bin/python -m pytest tests/unit/web/sessions/test_persist_compose_turn.py -v -k "insert_composition_state or session_write_lock_serializes_sqlite_same_session_state_version_allocation"
.venv/bin/python -m pytest tests/unit/web/sessions/ -v
```
Expected: PASS for the new tests AND every previously-passing sessions test continues to pass (the refactor must not break existing behaviour).

- [ ] **Step 5: Do not commit this helper standalone**

Keep the helper, `service.py` state-writer edits, and direct test-writer migrations with the schema/current-writer cutover. `_insert_composition_state` depends on `composition_states.provenance`, and the cutover is not reviewable until `tests/unit/web/sessions/test_models.py`, `tests/unit/web/blobs/test_service.py`, and `tests/unit/web/composer/test_tools.py` have been updated or routed through canonical row factories.

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

These behaviours exist in the current `add_message` implementation
and MUST survive the rewrite. Each has an
explicit regression test in this task.

1. **Cross-session guard** — `_assert_state_in_session` (module-level
   function in `service.py`) is invoked when `composition_state_id`
   is not `None`, raising `RuntimeError` if the state does not belong
   to the supplied session. This is the offensive-programming pattern
   CLAUDE.md mandates and the only protection against cross-session
   `composition_state_id` misuse before the FK fires.
2. **`updated_at` write-through** — `sessions_table.updated_at` is
   bumped to `now` on every successful insert. The list-sessions UI/API
   path orders sessions by this column; dropping the write
   silently misorders sessions on the home screen.
3. **`raw_content` persistence** — when `raw_content` is supplied
   (assistant messages whose visible `content` was rewritten by
   runtime preflight redaction), the value is stored verbatim in the
   `raw_content` column. This is the audit-attribution data
   referenced in spec §2 and produced by routes 1749 and 2152.
4. **`ChatMessageRecord` return** — the method returns a populated
   `ChatMessageRecord` (with the freshly-allocated `id`, `created_at`,
   the supplied `raw_content`/`composition_state_id`,
   `writer_principal`, and the new `tool_call_id` /
   `parent_assistant_id` linkage fields). Route
   callers that read `assistant_msg.id` and tests across `test_fork.py`,
   `test_routes.py`, and `test_service.py`
   consume the returned record.

### 14.3 Files in the atomic commit

This single commit modifies:

- `src/elspeth/web/sessions/protocol.py` — `ChatMessageRole`, `ChatMessageRecord`, and `SessionServiceProtocol.add_message` declaration (re-grep the current file before editing).
- `src/elspeth/web/sessions/service.py` — `SessionServiceImpl.add_message` body and helper internals, **plus `fork_session`'s direct batch insert into `chat_messages_table`** (see §14.6 below — this is an additional production writer that is invisible to the `\.add_message(` grep below).
- `src/elspeth/web/sessions/routes.py` — 6 production `service.add_message(...)` call sites, including `_persist_tool_invocations` and `_persist_llm_calls`.
- `evals/lib/decode_tools.py` — raw SQLite diagnostic reader for
  `chat_messages`; switch rev-4 DB reads from `ORDER BY created_at, id`
  to `ORDER BY sequence_no`.
- `tests/unit/web/sessions/test_protocol.py` — protocol/dataclass coverage for the new `audit` role plus `writer_principal` and message linkage fields.
- `tests/unit/web/sessions/test_service.py` — ~10 call sites.
- `tests/unit/web/sessions/test_fork.py` — ~38 call sites **plus a new regression test for the `fork_session` batch-insert sweep — see §14.6**.
- `tests/unit/web/sessions/test_routes.py` — ~22 call sites.
- `tests/unit/web/sessions/test_datetime_timezone.py` — 1 call site.
- `tests/unit/web/sessions/test_persist_compose_turn.py` — extends with `add_message` regression tests in this task.
- `tests/unit/evals/lib/test_decode_tools.py` — update the standalone
  `chat_messages` fixture schema and raw inserts so they mirror the new
  required columns (`sequence_no`, `writer_principal`, `tool_call_id`,
  `parent_assistant_id`) or deliberately document why any direct fixture
  field remains absent. This file is not an `add_message` caller, but it
  is a raw chat-message writer and must move in the same schema/current-
  writer cutover.

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

Also add a `fork_session` response backstop: copied `role="audit"` rows may be present in the source slice for audit fidelity, but they must not appear in any user-facing `new_messages`/fork response payload. Either filter them while building the returned `new_messages` list or prove with a regression that the source slice cannot include audit rows. Required test: fork a session that contains an audit row and assert the DB copy keeps it while the fork response excludes it.

### 14.5 Test-suite call-site migration

The exact number of `service.add_message(...)` test calls drifts as the
suite grows, and the grep below is authoritative. Most current test
callers are positional `service.add_message(session.id, "user", "msg")`
patterns. The mechanical migration: append
`, writer_principal=<value>` to every call, choosing the value by role:

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

- [ ] **Step 2: Write the regression tests pinning the preserved behaviours**

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
    assert result.writer_principal == "route_user_message"
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

Append the new metadata/linkage fields to `ChatMessageRecord` so every
read path hydrates the columns Task 1 added:

```python
    writer_principal: str
    tool_call_id: str | None = None
    parent_assistant_id: UUID | None = None
```

`writer_principal` is required on every hydrated record because
`fork_session` must preserve copied history's original audit writer
instead of deriving one from role. `tool_call_id` is the
provider/tool-call identifier and remains a string.
`parent_assistant_id` is a chat-message primary key exposed by the
service layer the same way `id`, `session_id`, and
`composition_state_id` are exposed: as `UUID` values, not raw DB strings.

Then replace the existing declaration with:

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

In `src/elspeth/web/sessions/service.py`, update the protocol imports to include `ChatMessageRole` alongside `ChatMessageRecord`/`CompositionStateData` before using it in the concrete `add_message` signature. In `src/elspeth/web/sessions/routes.py`, import `ChatMessageRole` from `elspeth.web.sessions.protocol` before using it in `_persist_tool_invocations`.

- [ ] **Step 5: Rewrite `SessionServiceImpl.add_message`**

In `src/elspeth/web/sessions/service.py`, replace the existing
`add_message` body with:

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
            # module-level function in service.py, NOT a method on self.
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
        writer_principal=writer_principal,
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
    """Persist tool/audit breadcrumbs without violating role=tool parent invariants.

    Audit-breadcrumb persistence remains fail-soft for SQLAlchemyError:
    a sidecar audit-row insert failure must not mask the primary composer
    outcome that the caller is already returning. Preserve the existing
    class-name-only structured logging; do not log SQL text, params, row
    payloads, raw exception strings, or chained exception text.
    """
    for invocation in tool_invocations:
        # ... existing content selection ...
        role: ChatMessageRole = "tool" if parent_assistant_id is not None else "audit"
        try:
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
        except SQLAlchemyError as save_err:
            slog.error(
                "composer_tool_invocation_persist_failed",
                session_id=str(session_id),
                tool_call_id=invocation.tool_call_id,
                tool_name=invocation.tool_name,
                exc_class=type(save_err).__name__,
            )


async def _persist_llm_calls(
    service: SessionServiceProtocol,
    session_id: UUID,
    llm_calls: tuple[ComposerLLMCall, ...],
    composition_state_id: UUID | None,
) -> None:
    """Persist provider-call audit breadcrumbs as internal audit rows.

    SQLAlchemyError from the audit sidecar write remains fail-soft and
    class-name-only, matching the pre-1A contract. The user-visible
    composer response must not fail only because an audit breadcrumb row
    could not be persisted.
    """
    for call in llm_calls:
        # ... existing content selection ...
        try:
            await service.add_message(
                session_id,
                "audit",
                content,
                tool_calls=[llm_call_audit_envelope(call)],
                composition_state_id=composition_state_id,
                writer_principal="compose_loop",
            )
        except SQLAlchemyError as save_err:
            slog.error(
                "composer_llm_call_persist_failed",
                session_id=str(session_id),
                model_requested=call.model_requested,
                status=call.status.value,
                exc_class=type(save_err).__name__,
            )
```

**Audit fail-soft regressions.** Update
`test_send_message_llm_call_persistence_failure_does_not_mask_success`
so its monkeypatch triggers on the new `role == "audit"` storage shape
with `_kind == "llm_call_audit"`; the existing pre-1A test triggers on
`role == "tool"` and would become a false green after the role split.
Add the symmetric tool-invocation persistence-failure regression for
`_persist_tool_invocations`: force `service.add_message(...)` to raise
`OperationalError` for either a parented `"tool"` audit breadcrumb or an
unparented `"audit"` breadcrumb, then assert the route still returns the
primary composer response and the structured log contains only the
exception class. These tests pin the logging/telemetry policy boundary:
audit sidecar persistence failures are observable but do not mask the
already-produced user-facing outcome.

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
- `tests/unit/web/sessions/test_protocol.py` (new/updated protocol coverage)
- `tests/unit/web/sessions/test_service.py` (~10 sites)
- `tests/unit/web/sessions/test_fork.py` (~38 sites)
- `tests/unit/web/sessions/test_routes.py` (~22 sites)
- `tests/unit/web/sessions/test_datetime_timezone.py` (1 site)

For each call, append `, writer_principal=<value>` per the role-keyed
default table. Hand-review any test whose intent is to exercise a
specific principal (none exist today; new ones may have been added
after this plan was authored).

`test_protocol.py` does not participate in the call-site migration by
volume, but it is part of the same sweep because `ChatMessageRole` and
`ChatMessageRecord` change in this atomic cut. Required assertions:
`ChatMessageRole` accepts `"audit"` at type-test/cast boundaries,
`ChatMessageRecord` exposes `tool_call_id` and `parent_assistant_id`,
and frozen message records preserve those scalar linkage values without
adding them to the `freeze_fields(self, "tool_calls")` container guard.

- [ ] **Step 8: Run the full sessions unit and integration suites**

```bash
.venv/bin/python -m pytest tests/unit/web/sessions/ tests/unit/web/blobs/test_routes.py tests/unit/web/blobs/test_service.py tests/unit/web/composer/test_tools.py -v
.venv/bin/python -m pytest tests/unit/evals/lib/test_decode_tools.py -v
.venv/bin/python -m mypy src/elspeth/web/sessions/service.py src/elspeth/web/sessions/protocol.py src/elspeth/web/sessions/routes.py
.venv/bin/python -m ruff check src/elspeth/contracts/advisory_locks.py src/elspeth/web/sessions/service.py src/elspeth/web/sessions/protocol.py src/elspeth/web/sessions/routes.py tests/unit/web/sessions/test_static_direct_writers.py tests/unit/web/sessions/test_protocol.py
.venv/bin/python scripts/cicd/enforce_tier_model.py check --root src/elspeth --allowlist config/cicd/enforce_tier_model
```
Expected: every 1A-owned test PASS; mypy clean; ruff clean; tier-model enforcement still passes after adding `contracts/advisory_locks.py` and new session-service imports. Do not include Schedule 1B-only integration regressions in 1A verification.
Existing `tests/integration/web/` coverage is still part of the merge
gate below; this sentence means 1A must not invent new 1B-only
compose-turn regressions as a prerequisite for the current-writer
cutover.

**Merge gate, not just local smoke.** The targeted commands above are
the fast developer loop for this task. Before merging the 1A cutover,
run the full web blast-radius gate because this schedule adds shared
`tests/unit/web/conftest.py` and `tests/integration/web/conftest.py`
fixtures:

```bash
.venv/bin/python -m pytest tests/unit/web/ tests/integration/web/ -v
git diff --name-only -- '*.py' | xargs -r .venv/bin/python -m ruff check
```

The second command intentionally lints every changed Python file,
including new conftests and tests outside `tests/unit/web/sessions/`.
Passing only the narrower sessions command is not sufficient evidence
for merge.

- [ ] **Step 9: Single atomic commit**

```bash
git add src/elspeth/web/sessions/service.py \
        src/elspeth/web/sessions/models.py \
        src/elspeth/web/sessions/protocol.py \
        src/elspeth/web/sessions/routes.py \
        src/elspeth/contracts/advisory_locks.py \
        tests/unit/web/sessions/test_static_direct_writers.py \
        tests/unit/web/sessions/test_chat_messages.py \
        tests/unit/web/sessions/test_composition_states.py \
        tests/unit/web/sessions/test_audit_access_log.py \
        tests/unit/web/sessions/test_persist_compose_turn.py \
        tests/unit/web/sessions/test_protocol.py \
        tests/unit/web/sessions/test_service.py \
        tests/unit/web/sessions/test_fork.py \
        tests/unit/web/sessions/test_routes.py \
        tests/unit/web/sessions/test_datetime_timezone.py \
        tests/unit/web/sessions/test_models.py \
        tests/unit/web/blobs/test_service.py \
        tests/unit/web/composer/test_tools.py \
        tests/unit/evals/lib/test_decode_tools.py
git commit -m "feat(sessions)!: rev-4 add_message rewrite — required writer_principal, sequence_no allocation, full call-site sweep (composer-progress-persistence phase 1)

BREAKING: SessionServiceProtocol.add_message and SessionServiceImpl.add_message
now require a keyword-only \`writer_principal\` argument matching the
\`ck_chat_messages_writer_principal\` CHECK enum. Adds optional
\`tool_call_id\` and \`parent_assistant_id\` arguments (required when
role='tool', forbidden otherwise — enforced by CHECK constraints).
Adds \`sequence_no\` allocation under _session_write_lock.

PRESERVED behaviours: _assert_state_in_session cross-session guard,
sessions.updated_at write-through, raw_content persistence,
ChatMessageRecord return type and writer_principal hydration. Each
preserved behaviour has an explicit regression test.

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
flagged that `fork_session` writes
`chat_messages` rows via a direct `conn.execute(insert(chat_messages_table), msg_records_data)`
batch, completely bypassing `add_message`. The plan's route-level
inventory already covers six `service.add_message(...)` writers; this
direct insert is the additional production `chat_messages` writer that
the `.add_message(` grep cannot see. After Task 1 lands NOT NULL `writer_principal`
and `sequence_no`, every `/fork_from_message` request would crash at
runtime with `IntegrityError`. This sweep brings `fork_session` into
the same atomic commit as the `add_message` rewrite.

**Prerequisite inside the atomic commit.** Complete the protocol and
read-path hydration edits from §14.4/§14.7 before coding this fork copy
sweep. `fork_session` needs `ChatMessageRecord.writer_principal`,
`ChatMessageRecord.tool_call_id`, and
`ChatMessageRecord.parent_assistant_id` populated by `get_messages`;
using those fields before protocol/dataclass/get-message hydration lands
turns the fork code into an attribute-error path instead of a schema
cutover proof. This is still one atomic commit, but the implementation
order matters: protocol + hydration first, then fork copying.

**File:** `src/elspeth/web/sessions/service.py` (`fork_session`; re-grep the current function before editing).

**Change 1 — populate truthful `writer_principal` for every row in `msg_records_data`.**

The fork transaction builds three categories of rows:

| Row category | Source line range | `writer_principal` value | Rationale |
|---|---|---|---|
| Copied source-session messages | ~1115-1129 | `msg.writer_principal` from the hydrated source row | These rows came from the source session's history. 1A must preserve the stored audit writer; deriving a replacement from `role` fabricates provenance and is not acceptable for copied history. `ChatMessageRecord` therefore exposes `writer_principal` in this same atomic cut. |
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
            "writer_principal": msg.writer_principal,  # NEW, preserved
            "created_at": msg.created_at,
            "composition_state_id": None,  # Don't reference source session states
        }
    )
```

This requires `ChatMessageRecord` to expose `writer_principal`,
`tool_call_id`, and `parent_assistant_id` (all are post-Task-1
columns). Confirm at the top of T14 that the protocol dataclass has
been extended to include them — if not, this is the commit that extends
it (it's part of the same atomic cut).

The exact field declarations to append to
`src/elspeth/web/sessions/protocol.py`'s `ChatMessageRecord` dataclass
(end-of-class style — match the existing field placement for
`raw_content`, `tool_calls`, `composition_state_id`):

```python
    writer_principal: str
    tool_call_id: str | None = None
    parent_assistant_id: UUID | None = None
```

Type rationale: `writer_principal` is a required scalar because copied
fork history must preserve the original writer; no role-keyed fallback
is allowed for source rows. `tool_call_id` is a provider/tool-call
identifier and stays `str | None`. `parent_assistant_id` references
`chat_messages.id`, which the service layer already exposes as `UUID`
on `ChatMessageRecord.id`; hydrate it to `UUID | None` in every
constructor (`add_message`, `get_messages`, and `fork_session`
returns). **No `freeze_fields` call is required for these three fields**
— they are scalar/immutable values, so the `frozen=True` slot itself is
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

**No role-keyed fallback helper.** Do not add a
`_ROLE_TO_WRITER_PRINCIPAL` fallback for copied source rows. It would
silently rewrite audit metadata for any source row whose stored writer
differs from the default, including future/admin tooling. New synthetic
fork rows use the explicit `"session_fork"` value; copied rows use the
stored `msg.writer_principal`.

**Required regression tests** (add to `tests/unit/web/sessions/test_fork.py`):

1. `test_fork_session_preserves_copied_writer_principal` — fork a
   session that contains user, assistant, system, audit, and tool rows;
   assert each copied row's `writer_principal` exactly matches the
   source row's stored value, while the new system fork notice and new
   edited user message both have `writer_principal="session_fork"`.
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
5. `test_fork_session_preserves_admin_tool_writer_principal_on_copied_rows`
   — insert a reviewed direct source row with
   `writer_principal="admin_tool"` and verify the fork copy keeps that
   exact value. This is the regression that prevents future code from
   reintroducing role-keyed provenance fabrication.

All five tests are required Green-bar additions. Do not skip the
`admin_tool` preservation test: it can be built with a reviewed direct
row and the schema enum already allows the value.

---

### 14.7 `get_messages` ordering switch + `fork_session` microsecond-hack removal (closes B2)

**Why this section exists.** The Phase 1 plan-review synthesis (B2)
flagged that `get_messages` currently orders by
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

**Change 1 — `get_messages` ORDER BY clause.**

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
unique within a session. Treat density only as a per-successful-batch
expectation (for example, a single fork copy gets `[1, 2, ...]`);
`sequence_no` is not a global gap-free counter and rollback/cancel paths
must not rely on it being dense forever.

In the same `get_messages` edit, hydrate the new metadata/linkage fields
on every returned `ChatMessageRecord`:

```python
ChatMessageRecord(
    # ... existing fields ...
    writer_principal=row.writer_principal,
    tool_call_id=row.tool_call_id,
    parent_assistant_id=UUID(row.parent_assistant_id)
    if row.parent_assistant_id is not None
    else None,
)
```

This is not optional: §14.6's `fork_session` copy loop reads
`msg.writer_principal`, `msg.tool_call_id`, and
`msg.parent_assistant_id`, and Phase 3/4 recovery surfaces rely on the
service API carrying the same audit writer and parentage the DB
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

`rg -n "get_messages|chat_messages_table.*order_by|ORDER BY .*created_at|FROM chat_messages" src tests evals -g '*.py'`
to enumerate every consumer of `get_messages`. As of the rev-4
snapshot:

- `src/elspeth/web/sessions/routes.py:412` (`_composer_chat_history`)
  reads `get_messages` output to build the LLM input. Behaviour is
  unchanged — the order it already expects (assistant before its tool
	  rows, system before user, copied messages before fork-time inserts)
	  is exactly what `sequence_no` enforces.
- `evals/lib/decode_tools.py:55` reads the same SQLite session DB
  directly for audit/transcript diagnostics and currently orders by
  `created_at, id`. This is a raw SQL read-side consumer, not a service
  `get_messages` caller, and it MUST move to `ORDER BY sequence_no`
  in the same 1A cutover. Leaving it on `created_at, id` would make the
  diagnostic view disagree with the canonical service ordering exactly
  where 1A is trying to remove same-timestamp ambiguity.
- Phase 4 recovery panel (not yet implemented) will consume
  `get_messages` and depends on stable intra-turn ordering. The
  ordering switch is its prerequisite.

No call site is broken by the change; the switch is a strict
strengthening of the previously implicit (and on fast SQLite,
unreliable) `created_at` ordering.

**Required 1A regression test.** Add
`test_get_messages_orders_same_timestamp_rows_by_sequence_no` to
`tests/unit/web/sessions/test_service.py` or
`tests/unit/web/sessions/test_persist_compose_turn.py`. The test must
create a session, write two or more rows with the same `created_at`
timestamp but intentionally non-chronological `sequence_no` values
through the current 1A writer/helper path, call `get_messages`, and
assert the returned roles/ids are ordered by `sequence_no`, not
`created_at` or insertion accident. This keeps the ordering proof
inside Schedule 1A; the multi-tool `persist_compose_turn` integration
test remains Schedule 1B coverage.

**Required eval diagnostic regression.** Extend
`tests/unit/evals/lib/test_decode_tools.py` so the standalone fixture
schema mirrors the rev-4 `chat_messages` columns needed by the decoder:
`sequence_no`, `writer_principal`, `tool_call_id`, and
`parent_assistant_id`. Update raw inserts to supply `sequence_no` and
`writer_principal` for every row and linkage fields for tool rows when
present. Add `test_decode_tool_sequence_orders_same_timestamp_rows_by_sequence_no`:
create rows for one session with identical `created_at` values, insert
them in a non-sequence order, call `decode_tool_sequence`, and assert the
returned roles/ids follow `sequence_no`, not `created_at`, `id`, or
insertion accident. Run this test with the rest of Task 14's targeted
verification:

```bash
.venv/bin/python -m pytest tests/unit/evals/lib/test_decode_tools.py -v
```

---
## Task 18: Document the staging session-DB recreation procedure

**Execution gate:** Complete this task before Task 1 Step 3 metadata changes are committed. The staging reset procedure is part of the schema cutover safety proof, not cleanup after the fact.

**Why this task exists.** Phase 1 changes the `chat_messages` and
`composition_states` schemas. ELSPETH has no Alembic migration
framework for the web session DB, and
`src/elspeth/web/sessions/schema.py` validates existing schema shape
instead of altering tables in place. Row-level `DELETE FROM
chat_messages` / `DELETE FROM composition_states` is therefore
incorrect: it leaves the old table shape behind and startup rejects the
stale DB.

The correct pre-release procedure is to stop the service, verify no
writer still has the SQLite database open, archive the current session
DB artifact set, remove the live artifact set, and restart so
`initialize_session_schema` recreates the schema from current metadata.
For SQLite, the artifact set is not just `sessions.db`: the main file,
`sessions.db-wal`, `sessions.db-shm`, and `sessions.db-journal` are one
rollback/recreate unit. Never archive or delete only the main file.

**Files:**
- Modify: `docs/runbooks/staging-session-db-recreation.md` (existing canonical reset runbook)
- Optional create: `docs/runbooks/staging-session-db-recreation.md` only if the repo's runbook index requires a staging-specific wrapper; if created, it must link back to `docs/runbooks/staging-session-db-recreation.md` and not fork the stop/go gates.

- [ ] **Step 1: Extend the existing session DB reset runbook**

Start from `docs/runbooks/staging-session-db-recreation.md`. Keep its Landscape orphaning stop/go gates, path-resolution rules, health checks, and create-session/journal verification. Add any 1A-specific schema-cutover notes there first; do not create a second reset procedure that can drift from the existing guide.

Update the guide's expected session-table inventory in the same edit so
it includes the new `audit_access_log` table. `initialize_session_schema`
validates the metadata table set exactly; a reset guide that still lists
only the pre-1A tables is stale and cannot be used as cutover evidence.

- [ ] **Step 2: Add the session-DB recreation procedure**

If the repo needs a staging-specific wrapper, create
`docs/runbooks/staging-session-db-recreation.md` with this section and
link it to `docs/runbooks/staging-session-db-recreation.md`. If any existing staging
note references row-level DELETE SQL or `elspeth migrate up` for the
session DB, replace it.

````markdown
# Staging Session DB Recreation

ELSPETH staging (`elspeth.foundryside.dev`) serves the source checkout at
`/home/john/elspeth` through `elspeth-web.service`. The web session DB
has no Alembic migrations. When a pre-release plan changes the session
schema, recreate the DB file from current metadata.

This procedure destroys staging session rows, chat history, composition
states, audit access log rows, runs, run events, blob/blob-link database
records, and encrypted `user_secrets` stored in the web session DB. It
does not delete blob payload files under the data directory, Landscape
audit data, payload storage, Filigree state, or source files. Do not run
it outside staging.
For SQLite, `sessions.db`, `sessions.db-wal`, `sessions.db-shm`, and
`sessions.db-journal` are handled as one matched artifact set for
archive, deletion, and rollback.

## Preconditions

1. The host is the staging host for `elspeth.foundryside.dev`.
2. No human operator is mid-session.
3. The source checkout at `/home/john/elspeth` is on the commit being
   deployed.
4. `deploy/elspeth-web.env` has been inspected directly for session DB
   settings without printing secret values.
5. The stop/go gates in `docs/runbooks/staging-session-db-recreation.md` have been run:
   Landscape code/schema must not reference web-session identifiers.
6. The pre-cutover source ref compatible with the archived DB has been
   recorded. If rollback is needed, restore that ref and the archived DB
   together; never run the old DB under the new schema code.
7. The live SQLite deployment is single-worker: `deploy/elspeth-web.service`
   has no `--workers` flag, `WEB_CONCURRENCY` is unset or `1`, and the
   startup multi-worker guard in `src/elspeth/web/app.py` remains enabled.
8. The operator has explicitly signed off on the `user_secrets` blast
   radius. Either the archived DB is the accepted recovery point, or
   staging secrets have a documented re-entry/reseed procedure before
   users resume composer work.
9. No other host-side process is writing the SQLite DB. The procedure
   stops `elspeth-web.service` and checks open handles before copying;
   if another process still has the main DB or a sidecar open, stop and
   identify it before continuing.

## Procedure

```bash
set -euo pipefail

PROJECT_ROOT="/home/john/elspeth"
ENV_FILE="$PROJECT_ROOT/deploy/elspeth-web.env"
SERVICE="elspeth-web.service"
PROJECT_ROOT_CANON="$(realpath -m "$PROJECT_ROOT")"

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
    /*) ;;
    *) DB_PATH="$PROJECT_ROOT/$DB_PATH" ;;
esac
DB_PATH="$(realpath -m "$DB_PATH")"

case "$DB_PATH" in
    "$PROJECT_ROOT_CANON"/*) ;;
    *) echo "REFUSING: DB_PATH is outside $PROJECT_ROOT_CANON: $DB_PATH" >&2; exit 1 ;;
esac

DB_ARTIFACTS=(
    "$DB_PATH"
    "$DB_PATH-wal"
    "$DB_PATH-shm"
    "$DB_PATH-journal"
)

echo "Resolved staging session DB path: $DB_PATH"
read -r -p "Archive and recreate this staging DB? Type RECREATE to continue: " CONFIRM
if [ "$CONFIRM" != "RECREATE" ]; then
    echo "Aborted."
    exit 1
fi

sudo systemctl stop "$SERVICE"

if command -v fuser >/dev/null 2>&1; then
    for artifact in "${DB_ARTIFACTS[@]}"; do
        if [ -e "$artifact" ] && sudo fuser "$artifact" >/dev/null 2>&1; then
            echo "REFUSING: $artifact is still open after $SERVICE stopped." >&2
            sudo fuser -v "$artifact" >&2 || true
            exit 1
        fi
    done
fi

FOUND_DB_ARTIFACT=0
for artifact in "${DB_ARTIFACTS[@]}"; do
    if [ -e "$artifact" ]; then
        FOUND_DB_ARTIFACT=1
    fi
done

if [ "$FOUND_DB_ARTIFACT" -eq 1 ]; then
    SNAPSHOT_DIR="$DB_PATH.pre-phase1.$(date -u +%Y%m%dT%H%M%SZ)"
    sudo mkdir -p "$SNAPSHOT_DIR"
    for artifact in "${DB_ARTIFACTS[@]}"; do
        if [ -e "$artifact" ]; then
            sudo cp -a "$artifact" "$SNAPSHOT_DIR/$(basename "$artifact")"
        fi
    done
    echo "Archived existing DB artifact set to $SNAPSHOT_DIR"
fi

for artifact in "${DB_ARTIFACTS[@]}"; do
    sudo rm -f "$artifact"
done
sudo systemctl start "$SERVICE"

curl --unix-socket /run/elspeth/uvicorn.sock -fsS http://localhost/api/health
curl -fsS https://elspeth.foundryside.dev/api/health
sudo systemctl status "$SERVICE" --no-pager --lines=20
```

`initialize_session_schema()` recreates the file on service startup.
If either health check fails, inspect `journalctl -u elspeth-web.service
--no-pager -n 80` before retrying.

After health checks pass, create a new session through the API or UI and
confirm no `SessionSchemaError` appears in the service journal. This is
the existing `docs/runbooks/staging-session-db-recreation.md` post-reset gate and must
remain in the staging wrapper if one is created.

Before handing staging back to users, verify the `user_secrets` outcome
the operator chose in the preconditions. If secrets were intentionally
cleared, confirm the affected composer/provider flow reports the expected
missing-secret state and that the operator has re-entered or reseeded any
required staging secrets. If rollback depends on the archive, confirm
the archived DB is retained because it contains the pre-reset encrypted
secret rows as well as chat/session data.

## Rollback

Rollback is allowed only before users resume work on the recreated DB, or
after explicitly preserving both the failed new DB and the archived old
DB for operator review. If the new service has accepted user traffic,
do not overwrite the new DB with the archive without a data-preservation
decision.

Rollback must restore source and data as a compatible pair. Because the
session DB also stores encrypted `user_secrets`, restoring the archive
also restores the pre-cutover staging secrets; never mix a new-source
checkout with the old secret/schema archive.

```bash
set -euo pipefail

PROJECT_ROOT="/home/john/elspeth"
SERVICE="elspeth-web.service"
DB_PATH="/absolute/path/resolved/by/the/procedure"
SNAPSHOT_DIR="/absolute/path/to/sessions.db.pre-phase1.YYYYMMDDTHHMMSSZ"
PRE_CUTOVER_REF="<recorded commit/ref compatible with SNAPSHOT_DIR>"
DB_ARTIFACTS=(
    "$DB_PATH"
    "$DB_PATH-wal"
    "$DB_PATH-shm"
    "$DB_PATH-journal"
)

sudo systemctl stop "$SERVICE"

FOUND_NEW_ARTIFACT=0
for artifact in "${DB_ARTIFACTS[@]}"; do
    if [ -e "$artifact" ]; then
        FOUND_NEW_ARTIFACT=1
    fi
done

if [ "$FOUND_NEW_ARTIFACT" -eq 1 ]; then
    FAILED_NEW_DB_DIR="$DB_PATH.failed-phase1.$(date -u +%Y%m%dT%H%M%SZ)"
    sudo mkdir -p "$FAILED_NEW_DB_DIR"
    for artifact in "${DB_ARTIFACTS[@]}"; do
        if [ -e "$artifact" ]; then
            sudo cp -a "$artifact" "$FAILED_NEW_DB_DIR/$(basename "$artifact")"
        fi
    done
    echo "Preserved failed new DB artifact set at $FAILED_NEW_DB_DIR"
fi

if [ -n "$(git -C "$PROJECT_ROOT" status --porcelain)" ]; then
    echo "REFUSING: $PROJECT_ROOT has uncommitted changes; preserve or commit them before rollback." >&2
    git -C "$PROJECT_ROOT" status --short >&2
    exit 1
fi

# Use the approved source-checkout rollback mechanism for staging. The
# important invariant is compatibility: the restored process must run the
# pre-cutover code that understands SNAPSHOT_DIR's schema. Do not use
# `git reset --hard` from automation; the dirty-tree guard above makes
# this fail closed before switching refs.
git -C "$PROJECT_ROOT" switch --detach "$PRE_CUTOVER_REF"

for artifact in "${DB_ARTIFACTS[@]}"; do
    sudo rm -f "$artifact"
done
for artifact in "${DB_ARTIFACTS[@]}"; do
    archived="$SNAPSHOT_DIR/$(basename "$artifact")"
    if [ -e "$archived" ]; then
        sudo cp -a "$archived" "$artifact"
    fi
done
sudo systemctl start "$SERVICE"

curl --unix-socket /run/elspeth/uvicorn.sock -fsS http://localhost/api/health
curl -fsS https://elspeth.foundryside.dev/api/health
sudo journalctl -u "$SERVICE" --no-pager -n 80
```

If rollback still fails, keep the service stopped, preserve both DB
files, and inspect the journal before trying another reset. Retrying the
new schema is appropriate when the failure was operational (wrong path,
permission, stale process). Rolling back is appropriate when the new code
or metadata is defective and the old DB snapshot must be served again.

## Why DELETE Is Forbidden

Row deletion does not add columns, CHECK constraints, FKs, or indexes.
For schema-changing Phase plans, row deletion leaves a stale DB shape
that the startup schema validator correctly rejects. The validator is
primarily a name/shape guard for expected tables, columns, CHECK names,
and index names; it is not a compatibility migration engine and must not
be treated as proof that stale CHECK expressions, partial-index
predicates, or old table layouts are safe. Archive/delete/recreate is
the only accepted cutover path for this 1A schema change. Archive,
delete, and rollback must handle the SQLite main DB plus `-wal`, `-shm`,
and `-journal` sidecars as a single artifact set; mixing a new main file
with stale sidecars, or restoring a main file without its sidecars, is
not a valid reset.
````

- [ ] **Step 3: Verify the procedure locally before committing**

Do a dry run against a throwaway copy of `data/sessions.db` or a
temporary local config path. Do not run the destructive staging command
from the Codex sandbox unless host/systemd access is explicitly
available. Record in the Phase 1 PR description:

- resolved DB path logic tested for default `data/sessions.db`
- resolved DB path logic tested for explicit relative `sqlite:///data/sessions.db`
- archive path format
- health-check command used for local smoke
- post-reset secret-dependent composer flow checked or explicitly
  recorded as waiting on operator secret re-entry/reseed

- [ ] **Step 4: Commit before schema/current-writer cutover**

```bash
git add docs/runbooks/staging-session-db-recreation.md
# If a staging-specific wrapper was created, stage it too:
# git add docs/runbooks/staging-session-db-recreation.md
git commit -m "docs(runbooks): session-DB recreation procedure for staging schema changes (composer-progress-persistence phase 1)"
```

---

---

## Schedule 1A Done When

1. [ ] The direct-write inventory for `chat_messages_table` and `composition_states_table` is captured in the PR body.
2. [ ] `tests/unit/web/sessions/test_static_direct_writers.py` fails closed on new SQLAlchemy or raw-SQL direct writers outside the reviewed allowlist.
3. [ ] No direct writer can insert a chat message or composition state without satisfying the new required columns.
4. [ ] `add_message` preserves cross-session guards, `updated_at`, `raw_content`, and `ChatMessageRecord` return hydration.
5. [ ] `fork_session` no longer bypasses sequence/provenance requirements and preserves copied rows' stored `writer_principal` values instead of deriving them from role.
6. [ ] Public route responses and composer prompt history exclude internal `role="audit"` rows.
7. [ ] Audit breadcrumb persistence failures remain fail-soft and class-name-only; they do not mask the primary composer response.
8. [ ] The staging session-DB recreation runbook exists, includes `audit_access_log` in the table/blast-radius text, restores DB + compatible source ref together, and fail-closes rollback on a dirty checkout before any source ref switch.
9. [ ] SQLite current-behavior tests pass.
10. [ ] 1A is documented as proving SQLite-current deployability only; full PostgreSQL DDL and concurrency proof belongs to Schedule 1C and is not claimed here.
11. [ ] Ruff, mypy, and tier-model enforcement pass for the 1A touched surfaces, and the merge gate includes `.venv/bin/python -m pytest tests/unit/web/ tests/integration/web/ -v`.
12. [ ] A follow-up review confirms Schedule 1A no longer blocks Schedule 1B.
