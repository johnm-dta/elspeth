# Phase 1A — Backend: `user_preferences` table + composer-preferences API

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land the backend half of Phase 1 — a new `user_preferences_table` in the session schema, Pydantic models, a `PreferencesService`, and two REST routes (`GET /api/composer-preferences`, `PATCH /api/composer-preferences`). Frontend wiring is in the companion plan, [Phase 1B](13-phase-1b-frontend.md).

**Architecture:** Schema-then-service-then-routes. The new table lives on the shared `metadata` in `src/elspeth/web/sessions/models.py` — no separate metadata object. The service is a thin wrapper around SQLAlchemy Core operations. Routes are FastAPI-standard and reuse the existing `get_current_user` dependency from `auth/middleware.py`. Per `project_db_migration_policy`, schema change ≡ delete-the-DB; this plan documents the operator action exactly once at the end (Phase 1B's smoke task carries it out).

**Tech Stack:** SQLAlchemy Core, FastAPI, Pydantic v2, pytest.

**Sibling plan:** [13-phase-1b-frontend.md](13-phase-1b-frontend.md) — frontend wiring + opt-out surfaces + smoke deploy.

**Roadmap reference:** [00-implementation-roadmap.md](00-implementation-roadmap.md).

---

## Scope boundaries

**In scope:**
- New `user_preferences_table` on `sessions/models.py`'s shared `metadata`.
- Package `src/elspeth/web/preferences/` with `__init__.py`, `models.py`, `service.py`, `routes.py`.
- Two REST endpoints (GET, PATCH) under `/api/composer-preferences`, both auth-required.
- Service-layer upsert with partial-update semantics.
- Tier-3 input validation via Pydantic `Literal["guided", "freeform"]`.
- Tier-1 read-side guards on the stored mode value.
- Wiring the new router into the FastAPI app composition site.

**Out of scope (handled in Phase 1B or later phases):**
- Frontend `preferencesStore`, `ComposerPreferences.tsx`, `UserMenu.tsx`, banners — all Phase 1B.
- Org-default override (Q F2 — out of scope).
- Tutorial-completed flag (Phase 4 will add a column).
- Telemetry preference-change event (Phase 8). Per CLAUDE.md primacy, every telemetry
  emission point must send or explicitly acknowledge "nothing to send." The PATCH route's
  success path carries the marker `# telemetry: deferred to Phase 8 polish —
  preference-change event`; Phase 8 will wire the emit.
- Existing-user migration heuristic — retired. Under `project_db_migration_policy` the
  sessions DB is deleted at deploy time, so the session-count heuristic always resolves
  to "new user → guided." The heuristic is removed from the service. See Task 5.

**Auditability boundary (CLAUDE.md attributability test):**
Composer mode is an authoring-time UI affordance, not a pipeline behaviour. Runtime
pipeline execution and its outputs do not vary with the mode the user used to compose.
The Landscape audit trail records pipeline runs; this preference does not affect any
auditable artifact and is therefore not recorded in the Landscape.

## Trust tier check (per CLAUDE.md)

- **Inbound preference value** (`PATCH` body): **Tier 3** — external. Pydantic `Literal["guided", "freeform"]` rejects anything else with a 422 at the boundary. No coercion: there is no "fix the typo" interpretation of `"kiosk"`.
- **Outbound preference value** (read from `user_preferences_table`): **Tier 1** — full trust. If the stored `default_composer_mode` is anything outside the literal, the service **crashes** with a `RuntimeError` containing the offending value and the user id. We control writes; a stored `"kiosk"` means tampering, corruption, or a code bug — never a recoverable situation.
- **Infrastructure failures** (DB unavailable, connection error): these are **not** Tier-1 corruption events. They propagate as unhandled exceptions, and FastAPI returns 500. The audit trail is unaffected; no special recovery logic is warranted at the route layer.

## File structure

**New:**

- `src/elspeth/web/preferences/__init__.py` — package marker (empty).
- `src/elspeth/web/preferences/models.py` — Pydantic request/response models.
- `src/elspeth/web/preferences/service.py` — `PreferencesService`.
- `src/elspeth/web/preferences/routes.py` — `create_preferences_router()`.
- `tests/unit/web/preferences/__init__.py` — package marker for tests.
- `tests/unit/web/preferences/test_schema.py` — schema-presence tests.
- `tests/unit/web/preferences/test_models.py` — Pydantic-model tests.
- `tests/unit/web/preferences/test_service.py` — service-layer tests.
- `tests/integration/web/test_preferences_routes.py` — FastAPI route tests.

**Modified:**

- `src/elspeth/web/sessions/models.py` — add `user_preferences_table` to the existing shared `metadata`.
- The FastAPI app-composition site — instantiate `PreferencesService` on `app.state` and `include_router(create_preferences_router())`. The exact file is identified during Task 4 (search via `grep -rn "create_session_router()" src/elspeth/web --include="*.py"`); the executor must read it before editing.

**Not modified in this phase:**
- `src/elspeth/web/auth/models.py` — `UserIdentity`/`UserProfile` remain unchanged. Preferences are keyed by `user_id` but not promoted into the identity dataclass.
- `src/elspeth/web/sessions/service.py` — session creation still does not write a mode field. The "default mode" is read on the frontend at create time.

## Database migration note (operator action — performed in Phase 1B)

This phase adds a table to a schema that has no Alembic. Per `project_db_migration_policy`,
**after merging this plan, the operator deletes the existing sessions database before restarting
the web service.** That action is performed by the Phase 1B smoke task (Task 9 of 1B) so the
backend and frontend land together without an intermediate broken state.

If you ship Phase 1A independently (e.g., to a staging branch), you must perform the DB delete
yourself before testing.

## Verification approach

Each task is TDD-shaped: failing test, minimal implementation, passing test, commit. After all
tasks land, the Phase 1B smoke task exercises the routes end-to-end against the live frontend.

---

## Task 1: Schema — add `user_preferences_table`

**Files:**
- Create: `tests/unit/web/preferences/__init__.py` (empty file; needed for pytest discovery).
- Create: `tests/unit/web/preferences/test_schema.py`.
- Modify: `src/elspeth/web/sessions/models.py` — append a new `Table` definition to the shared `metadata`.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/web/preferences/__init__.py` as an empty file (one-liner: `# Package marker for preferences tests.`).

Create `tests/unit/web/preferences/test_schema.py`:

```python
"""Schema-level tests: the user_preferences table exists with the expected columns."""

from elspeth.web.sessions.models import metadata


def test_user_preferences_table_registered() -> None:
    """The user_preferences table is registered on the shared metadata."""
    assert "user_preferences" in metadata.tables


def test_user_preferences_table_columns() -> None:
    """The user_preferences table has the expected columns."""
    table = metadata.tables["user_preferences"]
    column_names = {c.name for c in table.columns}
    assert column_names == {
        "user_id",
        "default_composer_mode",
        "banner_dismissed_at",
        "updated_at",
    }


def test_user_preferences_user_id_is_primary_key() -> None:
    """user_id is the primary key (one row per user)."""
    table = metadata.tables["user_preferences"]
    pk_columns = {c.name for c in table.primary_key.columns}
    assert pk_columns == {"user_id"}


def test_default_composer_mode_has_server_default_guided() -> None:
    """The stored default for new rows is 'guided' even at the DB level."""
    table = metadata.tables["user_preferences"]
    column = table.c.default_composer_mode
    # Server default's `.arg` is "guided" when set via `server_default="guided"`.
    assert column.server_default is not None
    assert "guided" in str(column.server_default.arg)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/unit/web/preferences/test_schema.py -v`
Expected: FAIL with `KeyError: 'user_preferences'` (and `AssertionError`s on the dependent assertions).

- [ ] **Step 3: Add the table definition**

In `src/elspeth/web/sessions/models.py`, after the existing table definitions, add:

```python
user_preferences_table = Table(
    "user_preferences",
    metadata,
    # One row per user. user_id is opaque (matches sessions_table.user_id);
    # we don't FK because auth providers can vary across deployments and
    # the sessions table itself has no users table to FK against.
    Column("user_id", String, primary_key=True),
    # The user's choice of default mode for NEW sessions. Per-session mode
    # toggles live in chat panel state and do not touch this row. The
    # service rejects values outside {"guided", "freeform"} at the boundary;
    # if a stored value violates that, the read-side guard crashes.
    Column(
        "default_composer_mode",
        String,
        nullable=False,
        server_default="guided",
    ),
    # Tracks the "we changed the default" banner so it only fires once per
    # user. NULL = banner not yet dismissed; non-NULL = dismissed-at timestamp.
    Column("banner_dismissed_at", DateTime(timezone=True), nullable=True),
    Column("updated_at", DateTime(timezone=True), nullable=False),
)
```

The existing imports at the top of the file already cover `Table`, `Column`, `String`, `DateTime`, and `metadata` — no new imports needed.

Note: do not add anything to `sessions_table` itself. Sessions are not tagged with a mode at the DB level. Mode is a per-session frontend state initialised from the preference at session-create time.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/unit/web/preferences/test_schema.py -v`
Expected: PASS — all four assertions green.

- [ ] **Step 5: Commit**

```bash
git add tests/unit/web/preferences/__init__.py tests/unit/web/preferences/test_schema.py src/elspeth/web/sessions/models.py
git commit -m "$(cat <<'EOF'
feat(web): add user_preferences table to session schema

Phase 1A.1 of composer UX redesign. Adds a one-row-per-user table holding
the default composer mode and the 'default changed' banner-dismissal
timestamp. Schema lives on the shared metadata in sessions/models.py.

Operator action required at full Phase 1 deploy time: delete the existing
sessions DB (per project_db_migration_policy — no Alembic). Phase 1B's
smoke task performs this step; do not ship 1A independently to a long-
lived environment without performing the delete.

See docs/composer/ux-redesign-2026-05/12-phase-1a-backend.md.
EOF
)"
```

## Task 2: Pydantic models for the preferences API

**Files:**
- Create: `src/elspeth/web/preferences/__init__.py` (empty file; one-line comment fine).
- Create: `src/elspeth/web/preferences/models.py`.
- Create: `tests/unit/web/preferences/test_models.py`.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/web/preferences/test_models.py`:

```python
"""Tests for the ComposerPreferences Pydantic models."""

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from elspeth.web.preferences.models import (
    ComposerPreferences,
    UpdateComposerPreferencesRequest,
)


def test_composer_preferences_valid() -> None:
    """A well-formed payload constructs cleanly."""
    payload = ComposerPreferences(
        default_mode="guided",
        banner_dismissed_at=None,
        updated_at=datetime.now(UTC),
    )
    assert payload.default_mode == "guided"


def test_composer_preferences_rejects_invalid_mode() -> None:
    """Tier-3 boundary: only 'guided' or 'freeform' are accepted."""
    with pytest.raises(ValidationError):
        ComposerPreferences(
            default_mode="kiosk",  # type: ignore[arg-type]
            banner_dismissed_at=None,
            updated_at=datetime.now(UTC),
        )


def test_update_request_accepts_full_payload() -> None:
    payload = UpdateComposerPreferencesRequest(default_mode="freeform")
    assert payload.default_mode == "freeform"
    assert payload.banner_dismissed_at is None


def test_update_request_accepts_only_banner_field() -> None:
    """Partial PATCH: caller sets only banner_dismissed_at."""
    stamp = datetime.now(UTC)
    payload = UpdateComposerPreferencesRequest(banner_dismissed_at=stamp)
    assert payload.default_mode is None
    assert payload.banner_dismissed_at == stamp


def test_update_request_rejects_invalid_mode() -> None:
    with pytest.raises(ValidationError):
        UpdateComposerPreferencesRequest(default_mode="kiosk")  # type: ignore[arg-type]


def test_update_request_accepts_empty_payload_as_noop() -> None:
    """An empty PATCH payload is a no-op; the request succeeds without changes."""
    payload = UpdateComposerPreferencesRequest()
    assert payload.default_mode is None
    assert payload.banner_dismissed_at is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/unit/web/preferences/test_models.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'elspeth.web.preferences'`.

- [ ] **Step 3: Implement the package and models**

Create `src/elspeth/web/preferences/__init__.py`:

```python
"""Composer-preferences package: per-user settings for the Web Composer."""
```

Create `src/elspeth/web/preferences/models.py`:

```python
"""Pydantic models for the composer-preferences API.

`ComposerPreferences` is the full response payload for GET; it is also the
response for PATCH (the service returns the post-write state). The PATCH
request body is `UpdateComposerPreferencesRequest`, which is a partial form
where each field is independently optional.
"""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict

ComposerMode = Literal["guided", "freeform"]


class ComposerPreferences(BaseModel):
    """The full preferences payload returned by GET and PATCH.

    Uses ``strict=True, extra="forbid"`` — the codebase convention for
    ``web/`` Pydantic models (see ``secrets/schemas.py``, ``blobs/schemas.py``,
    ``composer/progress.py``). ``strict=True`` rejects implicit coercions
    (e.g. ``"true"`` → ``True``) at the Tier-3 boundary; ``extra="forbid"``
    rejects unknown keys.

    Not declared ``frozen=True`` — existing ``web/`` models do not freeze,
    and FastAPI's response-model serialisation does not require it.
    """

    model_config = ConfigDict(strict=True, extra="forbid")

    default_mode: ComposerMode
    banner_dismissed_at: datetime | None
    updated_at: datetime


class UpdateComposerPreferencesRequest(BaseModel):
    """Partial-update payload for PATCH.

    Every field is independently optional; the service writes only the
    fields the caller actually set. An empty PATCH is a no-op (the request
    succeeds; `updated_at` is bumped if any row already exists).

    Same ``strict=True, extra="forbid"`` configuration as
    ``ComposerPreferences`` above — consistent with the codebase's web
    Pydantic models. ``extra="forbid"`` also ensures a typo in a field name
    (e.g. ``default_modd``) surfaces as a 422 rather than silently doing
    nothing.
    """

    model_config = ConfigDict(strict=True, extra="forbid")

    default_mode: ComposerMode | None = None
    banner_dismissed_at: datetime | None = None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/unit/web/preferences/test_models.py -v`
Expected: PASS — 6 tests green.

- [ ] **Step 5: Commit**

```bash
git add src/elspeth/web/preferences/__init__.py src/elspeth/web/preferences/models.py tests/unit/web/preferences/test_models.py
git commit -m "feat(web): add ComposerPreferences Pydantic models (Phase 1A.2)"
```

## Task 3: PreferencesService — read/write the table

**Files:**
- Create: `src/elspeth/web/preferences/service.py`.
- Create: `tests/unit/web/preferences/test_service.py`.

The service supports both the basic `guided` default (new users) and the existing-user migration heuristic (freeform default for users with prior sessions). Task 5 extends this further; Task 3 ships the base read/write logic.

- [ ] **Step 1: Write the failing test**

**Test isolation convention:** Each test uses a distinct `user_id` matching its scenario
name (e.g., `"alice-get-default"`, `"alice-update-persist"`). This eliminates accidental
cross-test state sharing without the overhead of rebuilding the schema per test. The
`engine` / `service` fixtures are module-scoped (default pytest scope); isolation is via
disjoint user namespaces, not per-test teardown.

```python
"""Tests for PreferencesService."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

import pytest
from sqlalchemy import create_engine

from elspeth.web.preferences.models import UpdateComposerPreferencesRequest
from elspeth.web.preferences.service import PreferencesService
from elspeth.web.sessions.models import metadata, user_preferences_table


@pytest.fixture(scope="module")
def engine():
    eng = create_engine("sqlite+pysqlite:///:memory:")
    metadata.create_all(eng)
    yield eng
    eng.dispose()


@pytest.fixture(scope="module")
def service(engine):
    return PreferencesService(engine, now=lambda: datetime(2026, 5, 15, tzinfo=UTC))


def test_get_for_new_user_returns_guided_default(service):
    """A user with no row gets the server-side default 'guided'."""
    prefs = asyncio.run(service.get_composer_preferences("alice-get-default"))
    assert prefs.default_mode == "guided"
    assert prefs.banner_dismissed_at is None


def test_update_persists_and_round_trips(service):
    payload = UpdateComposerPreferencesRequest(default_mode="freeform")
    result = asyncio.run(service.update_composer_preferences("alice-update-persist", payload))
    assert result.default_mode == "freeform"
    prefs = asyncio.run(service.get_composer_preferences("alice-update-persist"))
    assert prefs.default_mode == "freeform"


def test_partial_update_only_touches_provided_fields(service):
    asyncio.run(
        service.update_composer_preferences(
            "alice-partial",
            UpdateComposerPreferencesRequest(default_mode="freeform"),
        )
    )
    stamp = datetime(2026, 5, 15, 12, 0, tzinfo=UTC)
    asyncio.run(
        service.update_composer_preferences(
            "alice-partial",
            UpdateComposerPreferencesRequest(banner_dismissed_at=stamp),
        )
    )
    prefs = asyncio.run(service.get_composer_preferences("alice-partial"))
    assert prefs.default_mode == "freeform"  # not reset by the partial update
    assert prefs.banner_dismissed_at == stamp


def test_users_are_isolated(service):
    asyncio.run(
        service.update_composer_preferences(
            "alice-isolated", UpdateComposerPreferencesRequest(default_mode="freeform")
        )
    )
    bob_prefs = asyncio.run(service.get_composer_preferences("bob-isolated"))
    assert bob_prefs.default_mode == "guided"


def test_corrupt_stored_mode_crashes_loudly(service, engine):
    """Tier-1 guard: a stored value outside the literal raises (no recovery)."""
    with engine.begin() as conn:
        conn.execute(
            user_preferences_table.insert().values(
                user_id="alice-corrupt",
                default_composer_mode="kiosk",  # corrupt value
                banner_dismissed_at=None,
                updated_at=datetime.now(UTC),
            )
        )
    with pytest.raises(RuntimeError, match="kiosk"):
        asyncio.run(service.get_composer_preferences("alice-corrupt"))


def test_patch_with_valid_mode_recovers_from_corrupt_row(service, engine):
    """Corrupt-mode PATCH lockout must not occur (Finding 7).

    If a row has a corrupt mode and the caller sends a valid PATCH, the
    write must succeed and return the new valid mode. Before the fix,
    update_composer_preferences called get_composer_preferences after the
    write, which re-ran the Tier-1 guard against the (now-fixed) row.
    However, if payload.default_mode was None, the corrupt value stayed in
    the row and the guard crashed — even though the write itself succeeded.

    The fixed implementation returns from the values just written, not from
    a re-read. This test verifies a corrupt row + valid PATCH => clean response.
    """
    with engine.begin() as conn:
        conn.execute(
            user_preferences_table.insert().values(
                user_id="alice-corrupt-patch",
                default_composer_mode="kiosk",  # corrupt
                banner_dismissed_at=None,
                updated_at=datetime.now(UTC),
            )
        )
    # PATCH with an explicit valid mode: must succeed and return the new mode.
    result = asyncio.run(
        service.update_composer_preferences(
            "alice-corrupt-patch",
            UpdateComposerPreferencesRequest(default_mode="guided"),
        )
    )
    assert result.default_mode == "guided"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/unit/web/preferences/test_service.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'elspeth.web.preferences.service'`.

- [ ] **Step 3: Implement the service**

Create `src/elspeth/web/preferences/service.py`:

```python
"""Service layer for the user_preferences table.

Read path: returns the user's row; falls back to 'guided' when no row exists.
Crashes on Tier-1 read of a corrupt mode value (any stored value outside
{"guided", "freeform"} is a code bug, DB corruption, or tampering — never
a recoverable situation).

Write path: upserts the row, touching only fields the caller actually set.
Returns the response model built from the values just written rather than
re-reading — this avoids the corrupt-mode PATCH lockout (a pre-existing
corrupt row would crash a re-read even when the write succeeded).

SQLite-dialect-specific (`ON CONFLICT ... DO UPDATE`) for now; this is the
deployed dialect.

Async/sync bridge: uses ``run_sync_in_worker`` from
``elspeth.web.async_workers`` (single-worker pool with cancellation drain).
See ``sessions/service.py:380`` for the canonical usage pattern.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.engine import Engine

from elspeth.web.async_workers import run_sync_in_worker
from elspeth.web.preferences.models import (
    ComposerPreferences,
    UpdateComposerPreferencesRequest,
)
from elspeth.web.sessions.models import user_preferences_table

_DEFAULT_MODE = "guided"
_VALID_MODES = frozenset({"guided", "freeform"})


def _utcnow() -> datetime:
    return datetime.now(UTC)


class PreferencesService:
    """Reads and writes per-user composer preferences."""

    def __init__(self, engine: Engine, *, now: Callable[[], datetime] = _utcnow) -> None:
        self._engine = engine
        self._now = now

    async def get_composer_preferences(self, user_id: str) -> ComposerPreferences:
        """Return the user's preferences, falling back to 'guided' if no row exists.

        Default policy:
          - No row => 'guided' (new-user default; existing-user session-count
            heuristic retired under project_db_migration_policy — see Task 5).
          - Row exists => use stored value; crash if stored value is corrupt.
        """

        def _sync() -> ComposerPreferences:
            with self._engine.connect() as conn:
                row = conn.execute(
                    select(user_preferences_table).where(
                        user_preferences_table.c.user_id == user_id
                    )
                ).first()
                if row is not None:
                    return self._row_to_prefs(row, user_id)

            # No row: return the new-user guided default. We do not write a
            # row here (lazy: avoid write traffic for users who never touch
            # preferences).
            return ComposerPreferences(
                default_mode=_DEFAULT_MODE,
                banner_dismissed_at=None,
                updated_at=self._now(),
            )

        return await run_sync_in_worker(_sync)

    def _row_to_prefs(self, row: object, user_id: str) -> ComposerPreferences:
        """Convert a DB row to the response model with a Tier-1 read guard.

        A stored mode outside the validated set is a fault we caused (bug,
        tampering, or DB corruption). Crash with the offending value
        named so the operator can diagnose.
        """
        mode = row.default_composer_mode  # type: ignore[union-attr]
        if mode not in _VALID_MODES:
            raise RuntimeError(
                f"user_preferences row for {user_id!r} has invalid "
                f"default_composer_mode={mode!r}"
            )
        return ComposerPreferences(
            default_mode=mode,
            banner_dismissed_at=row.banner_dismissed_at,  # type: ignore[union-attr]
            updated_at=row.updated_at,  # type: ignore[union-attr]
        )

    async def update_composer_preferences(
        self, user_id: str, payload: UpdateComposerPreferencesRequest
    ) -> ComposerPreferences:
        """Upsert the preferences row, touching only fields in payload.

        Empty payloads are accepted as no-ops (the request succeeds; if a
        row already exists, only `updated_at` advances).

        Returns the response model built directly from the written values —
        no round-trip read. This prevents the corrupt-mode PATCH lockout:
        a pre-existing corrupt `default_mode` row would crash a re-read
        even when the PATCH write succeeded and supplied a valid new mode.
        Since the values just written are validated and trusted, the guard
        is not re-run.

        # telemetry: deferred to Phase 8 polish — preference-change event.
        # A PATCH here is a user-preference-change event that belongs in
        # operational telemetry. Wiring deferred; no `telemetry.emit()` call
        # is made here. Phase 8 will add the emit once the telemetry helper
        # is stable.
        """
        now = self._now()

        def _sync() -> tuple[str, datetime | None]:
            """Returns (resolved_mode, resolved_banner_dismissed_at) after write."""
            with self._engine.begin() as conn:
                # Determine the mode to write on insert (NOT NULL column).
                # On conflict, only updated fields are changed.
                if payload.default_mode is not None:
                    insert_mode = payload.default_mode
                else:
                    existing_mode = conn.execute(
                        select(user_preferences_table.c.default_composer_mode).where(
                            user_preferences_table.c.user_id == user_id
                        )
                    ).scalar_one_or_none()
                    insert_mode = existing_mode if existing_mode is not None else _DEFAULT_MODE

                # For the banner field: if the caller didn't set it we need
                # to preserve the existing value. Read it now (same
                # transaction) so we can return the full post-write state
                # without a second round-trip.
                if payload.banner_dismissed_at is not None:
                    resolved_banner = payload.banner_dismissed_at
                else:
                    resolved_banner = conn.execute(
                        select(user_preferences_table.c.banner_dismissed_at).where(
                            user_preferences_table.c.user_id == user_id
                        )
                    ).scalar_one_or_none()

                values: dict[str, object] = {
                    "user_id": user_id,
                    "default_composer_mode": insert_mode,
                    "banner_dismissed_at": payload.banner_dismissed_at,
                    "updated_at": now,
                }
                stmt = sqlite_insert(user_preferences_table).values(**values)
                update_clause: dict[str, object] = {"updated_at": now}
                if payload.default_mode is not None:
                    update_clause["default_composer_mode"] = payload.default_mode
                if payload.banner_dismissed_at is not None:
                    update_clause["banner_dismissed_at"] = payload.banner_dismissed_at
                stmt = stmt.on_conflict_do_update(
                    index_elements=["user_id"], set_=update_clause
                )
                conn.execute(stmt)

            return insert_mode, resolved_banner

        written_mode, written_banner = await run_sync_in_worker(_sync)
        return ComposerPreferences(
            default_mode=written_mode,
            banner_dismissed_at=written_banner,
            updated_at=now,
        )
```

**Notes on implementation choices:**
- `_run_sync` is removed. Use `run_sync_in_worker` directly (canonical import from
  `elspeth.web.async_workers`; see `sessions/service.py:380` for the same pattern).
  No `# type: ignore` is needed — the generic `[**P, T]` typing preserves the return type.
- The existing-user migration heuristic (`sessions_table` count query) is retired.
  The service no longer imports `sessions_table`. See Task 5 for rationale.
- `update_composer_preferences` returns from the values just written, not from a
  re-read via `get_composer_preferences`. This eliminates the corrupt-mode PATCH lockout
  (Finding 7): a corrupt pre-existing `default_mode` would crash a re-read even after
  the PATCH succeeded and wrote a valid mode. The written values are known-valid; the
  Tier-1 guard need not run on them.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/unit/web/preferences/test_service.py -v`
Expected: PASS — 6 tests green (5 core + 1 corrupt-mode recovery).

- [ ] **Step 5: Commit**

```bash
git add src/elspeth/web/preferences/service.py tests/unit/web/preferences/test_service.py
git commit -m "feat(web): add PreferencesService for composer prefs (Phase 1A.3)"
```

## Task 4: REST routes — GET and PATCH `/api/composer-preferences`

**Files:**
- Create: `src/elspeth/web/preferences/routes.py`.
- Create: `tests/integration/web/test_preferences_routes.py`.
- Modify: the FastAPI app-composition site (identified via grep below).

- [ ] **Step 1: Identify the FastAPI app-composition site**

Before writing routes, find where the existing routers are wired up:

```bash
grep -rn "create_session_router\b" src/elspeth/web --include="*.py" | grep -v __pycache__ | grep -v test
```

The hit will be a single file (typically `src/elspeth/web/app.py` or `src/elspeth/web/__init__.py`). Read that file end-to-end to understand the engine attribute name (`app.state.session_engine` vs `app.state.engine` etc.) before writing Step 4 below.

- [ ] **Step 2: Write the failing route test**

The integration test file is self-contained: it declares its own fixtures modelled
directly on `tests/integration/web/conftest.py` (`composer_test_client`, lines 62–102).
There is no shared `client_with_user` / `client_anonymous` fixture — those do not exist.

The fixtures here build their own in-memory engine, call `initialize_session_schema`,
wire `app.state.preferences_service`, and override `get_current_user`. Three fixtures
are needed:

- `client_as_alice` — authenticated as `user_id="alice"`.
- `client_as_bob` — authenticated as `user_id="bob"` (required for cross-user isolation
  test).
- `client_anonymous` — auth override raises `HTTPException(status_code=401)`
  directly. The real `get_current_user` (`auth/middleware.py:38–39`) reads
  `app.state.auth_audit_recorder` and `app.state.settings` *before* checking
  the Authorization header, so leaving the real dependency in place against
  a bare `FastAPI()` would raise `AttributeError` → 500 rather than 401.
  The override is the correct way to assert route-layer auth-required
  behaviour without standing up the full auth surface.

Create `tests/integration/web/test_preferences_routes.py`:

```python
"""Integration tests for /api/composer-preferences.

Fixtures are self-contained (no shared conftest fixtures); modelled on
tests/integration/web/conftest.py:composer_test_client.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from sqlalchemy.pool import StaticPool

from elspeth.web.auth.middleware import get_current_user
from elspeth.web.auth.models import UserIdentity
from elspeth.web.preferences.routes import create_preferences_router
from elspeth.web.preferences.service import PreferencesService
from elspeth.web.sessions.engine import create_session_engine
from elspeth.web.sessions.schema import initialize_session_schema


def _make_app(user_id: str | None, tmp_path: Path) -> FastAPI:
    """Build a minimal FastAPI app wired for preferences tests."""
    engine = create_session_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    initialize_session_schema(engine)
    app = FastAPI()
    app.state.preferences_service = PreferencesService(engine)
    app.state.session_engine = engine

    if user_id is not None:
        identity = UserIdentity(user_id=user_id, username=user_id)

        async def _mock_user() -> UserIdentity:
            return identity

        app.dependency_overrides[get_current_user] = _mock_user
    else:
        # Anonymous: override get_current_user with a stub that raises
        # 401. The real get_current_user reads app.state.auth_audit_recorder
        # and app.state.settings before inspecting the Authorization header,
        # so leaving it in place against this bare FastAPI() would raise
        # AttributeError → 500. The 401 override is the correct way to
        # assert route-layer auth-required behaviour without standing up
        # the full auth surface.
        async def _unauthenticated() -> UserIdentity:
            raise HTTPException(status_code=401, detail="Not authenticated")

        app.dependency_overrides[get_current_user] = _unauthenticated

    app.include_router(create_preferences_router())
    return app


@pytest.fixture
def client_as_alice(tmp_path: Path) -> Iterator[TestClient]:
    yield TestClient(_make_app("alice", tmp_path))


@pytest.fixture
def client_as_bob(tmp_path: Path) -> Iterator[TestClient]:
    yield TestClient(_make_app("bob", tmp_path))


@pytest.fixture
def client_anonymous(tmp_path: Path) -> Iterator[TestClient]:
    # raise_server_exceptions left at the default (True): the 401 override
    # raises HTTPException, which FastAPI converts to a 401 response — no
    # server exception should propagate. If one does, the test should fail
    # loudly rather than silently 500.
    yield TestClient(_make_app(None, tmp_path))


# ---------------------------------------------------------------------------
# Route tests
# ---------------------------------------------------------------------------


def test_get_returns_guided_default_for_brand_new_user(client_as_alice: TestClient) -> None:
    response = client_as_alice.get("/api/composer-preferences")
    assert response.status_code == 200
    body = response.json()
    assert body["default_mode"] == "guided"
    assert body["banner_dismissed_at"] is None


def test_patch_updates_default_mode(client_as_alice: TestClient) -> None:
    response = client_as_alice.patch(
        "/api/composer-preferences",
        json={"default_mode": "freeform"},
    )
    assert response.status_code == 200
    assert response.json()["default_mode"] == "freeform"

    follow_up = client_as_alice.get("/api/composer-preferences")
    assert follow_up.json()["default_mode"] == "freeform"


def test_patch_persists_banner_dismissal(client_as_alice: TestClient) -> None:
    response = client_as_alice.patch(
        "/api/composer-preferences",
        json={"banner_dismissed_at": "2026-05-15T12:00:00Z"},
    )
    assert response.status_code == 200
    assert response.json()["banner_dismissed_at"] == "2026-05-15T12:00:00Z"


def test_patch_rejects_invalid_mode(client_as_alice: TestClient) -> None:
    response = client_as_alice.patch(
        "/api/composer-preferences",
        json={"default_mode": "kiosk"},
    )
    assert response.status_code == 422


def test_patch_rejects_unknown_field(client_as_alice: TestClient) -> None:
    """Extra fields must 422: a typo in the field name must not silently no-op.

    UpdateComposerPreferencesRequest uses extra="forbid" (consistent with
    the codebase's web/ Pydantic models — secrets/schemas.py,
    composer/progress.py, composer/redaction.py).
    """
    response = client_as_alice.patch(
        "/api/composer-preferences",
        json={"default_modd": "freeform"},  # typo
    )
    assert response.status_code == 422


def test_get_requires_auth(client_anonymous: TestClient) -> None:
    response = client_anonymous.get("/api/composer-preferences")
    assert response.status_code == 401


def test_patch_requires_auth(client_anonymous: TestClient) -> None:
    response = client_anonymous.patch(
        "/api/composer-preferences",
        json={"default_mode": "freeform"},
    )
    assert response.status_code == 401


def test_users_cannot_see_each_others_preferences(
    client_as_alice: TestClient,
    client_as_bob: TestClient,
) -> None:
    """Route-level cross-user isolation: alice's prefs are invisible to bob.

    A service-layer test (`test_users_are_isolated`) also covers this, but
    a route bug could leak across users while service tests stay green.
    """
    # Alice sets freeform.
    resp = client_as_alice.patch(
        "/api/composer-preferences",
        json={"default_mode": "freeform"},
    )
    assert resp.status_code == 200

    # Bob — on a separate engine/app — still sees the guided default.
    bob_resp = client_as_bob.get("/api/composer-preferences")
    assert bob_resp.status_code == 200
    assert bob_resp.json()["default_mode"] == "guided"


def test_db_unavailable_returns_500(tmp_path: Path) -> None:
    """Infrastructure failure → 500 (not a Tier-1 corruption event)."""
    from unittest.mock import MagicMock, AsyncMock
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    app = FastAPI()
    identity = UserIdentity(user_id="alice", username="alice")

    async def _mock_user() -> UserIdentity:
        return identity

    broken_service: MagicMock = MagicMock()
    broken_service.get_composer_preferences = AsyncMock(
        side_effect=Exception("DB connection refused")
    )
    app.state.preferences_service = broken_service
    app.dependency_overrides[get_current_user] = _mock_user
    app.include_router(create_preferences_router())

    client = TestClient(app, raise_server_exceptions=False)
    response = client.get("/api/composer-preferences")
    assert response.status_code == 500
```

- [ ] **Step 3: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/integration/web/test_preferences_routes.py -v`
Expected: FAIL — the route doesn't exist (404) or fixture not yet importable.

- [ ] **Step 4: Implement the routes**

Create `src/elspeth/web/preferences/routes.py`:

```python
"""FastAPI router for composer-preferences endpoints."""

from fastapi import APIRouter, Depends, Request

from elspeth.web.auth.middleware import get_current_user
from elspeth.web.auth.models import UserIdentity
from elspeth.web.preferences.models import (
    ComposerPreferences,
    UpdateComposerPreferencesRequest,
)


def create_preferences_router() -> APIRouter:
    router = APIRouter(prefix="/api/composer-preferences", tags=["preferences"])

    @router.get("", response_model=ComposerPreferences)
    async def get_preferences(
        request: Request,
        user: UserIdentity = Depends(get_current_user),  # noqa: B008
    ) -> ComposerPreferences:
        service = request.app.state.preferences_service
        return await service.get_composer_preferences(user.user_id)

    @router.patch("", response_model=ComposerPreferences)
    async def update_preferences(
        body: UpdateComposerPreferencesRequest,
        request: Request,
        user: UserIdentity = Depends(get_current_user),  # noqa: B008
    ) -> ComposerPreferences:
        service = request.app.state.preferences_service
        return await service.update_composer_preferences(user.user_id, body)

    return router
```

- [ ] **Step 5: Wire the service + router into the FastAPI app**

In the file identified in Step 1 (typically alongside `create_session_router()`), add:

```python
from elspeth.web.preferences.routes import create_preferences_router
from elspeth.web.preferences.service import PreferencesService

# Wire the service. Use the same engine handle the sessions service uses;
# the user_preferences_table lives on the same metadata.
app.state.preferences_service = PreferencesService(app.state.session_engine)

# Register the router alongside the existing routers:
app.include_router(create_preferences_router())
```

The exact engine attribute (`session_engine` vs `engine` vs `auth_session_engine`) must be confirmed by reading the file. Do not guess — pick whatever the sessions service uses.

If the app-composition site uses a factory pattern (e.g., `create_app(settings)` that returns an `app`), put both lines inside that factory at the same scope where the session router is included.

- [ ] **Step 6: Run integration tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/integration/web/test_preferences_routes.py -v`
Expected: PASS — all tests green.

- [ ] **Step 7: Run the full integration suite to catch regressions**

Run: `.venv/bin/python -m pytest tests/integration/web/ -v`
Expected: PASS — no existing integration tests should break. If any do, the new router wiring is interacting with something it shouldn't (e.g., duplicate route prefix). Diagnose and fix.

- [ ] **Step 8: Commit**

Replace `<app-file>` with the actual file path identified in Step 1.

```bash
git add src/elspeth/web/preferences/routes.py <app-file> tests/integration/web/test_preferences_routes.py
git commit -m "feat(web): expose composer-preferences GET/PATCH routes (Phase 1A.4)"
```

## Task 5: ~~Existing-user migration coverage~~ — Retired

**Decision:** The existing-user migration heuristic (freeform default for users with prior
sessions) is retired. Under `project_db_migration_policy`, Phase 1B Task 9 deletes the
sessions DB before the service restarts. At that point `sessions_table` is empty for every
user, so the count-query always resolves to "new user → guided default" — the heuristic
silently never fires.

**Resolution (option a):** Simplify the service to "guided always; freeform only on
explicit opt-out." This removes:
- the `_DEFAULT_MODE_EXISTING_USER` constant
- the session-count query in both read and write paths
- the `sessions_table` import from `service.py`

All Task 5 tests are retired alongside the heuristic. The
`test_new_user_no_row_no_sessions_defaults_to_guided` case is already covered by
`test_get_for_new_user_returns_guided_default` in Task 3.

**Note on finding 11 (deduplicate `_resolve_default_mode`):** That finding becomes moot
when the migration heuristic is retired — the count-query branch no longer exists in
either the read or write path.

No commit is needed for this task. The heuristic removal is part of Task 3's service
implementation above.

---

## What Phase 1A leaves the backend in

After Tasks 1–4 land (Task 5 retired — see above):

- `user_preferences_table` is on the shared metadata; `metadata.create_all(engine)` will create it on a fresh DB.
- Two REST endpoints (`GET /api/composer-preferences`, `PATCH /api/composer-preferences`) return the user's preferences. New users always receive `guided` as the default; existing-user detection via session-count was retired (see Task 5).
- The service contains both a Tier-1 read guard (crashes on corrupt stored mode) and a Tier-3 boundary check (Pydantic rejects invalid input with 422).
- PATCH returns the response from the values just written, not from a second round-trip read (eliminates the corrupt-mode PATCH lockout).
- No frontend code has changed; the new endpoints are not yet consumed.

Phase 1B picks up from here: frontend store, opt-out surfaces, banner, smoke deploy.

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| Operator deploys 1A without the corresponding DB delete | Task 1's commit message calls this out. The intent is to ship 1A + 1B together; if 1A lands first, the operator must perform the delete from `project_db_migration_policy` before the route can serve traffic. |
| Engine attribute name guessed wrong in Task 4 | Step 1 of Task 4 names the grep verbatim; Step 5 instructs "do not guess". |
| Pydantic `extra="forbid"` regression for other surfaces | `extra="forbid"` is added only to `UpdateComposerPreferencesRequest`, not globally. The codebase's existing `web/` models use this pattern; it is consistent. |
| SQLite-dialect-specific `ON CONFLICT` ties us to SQLite | The deployment is SQLite per the session schema; this is intentional. If a future phase migrates to Postgres, the same dialect call exists on Postgres (`postgresql.dialects.insert`); swap the import. |
| DB unavailable at route time | Infrastructure failures (DB unavailable, connection error) propagate as 500 — these are **not** Tier-1 corruption events. They are transient operational failures; the audit trail is unaffected. No special handling in the route layer; FastAPI's default 500 response is correct. |
| Concurrent PATCH race window | Two near-simultaneous PATCHes for the same user resolve to the last write via SQLite `ON CONFLICT DO UPDATE`; this is deterministic and intentional. No error is raised; the race window is accepted. No test is added for this behaviour — the acknowledgment is the mitigation. |

## Forward compatibility

Under `project_db_migration_policy`, the sessions DB is deleted whenever the schema changes.
Subsequent phases that add columns to `user_preferences_table` (Phase 4 adds
`tutorial_completed_at`; Phase 8 may add telemetry flags) will wipe `banner_dismissed_at`
and any other Phase 1A-era user state on every deploy until a migration runner exists.
The structural fix (Alembic or equivalent) is owned by the roadmap; this plan notes the
dependency but does not implement it.

## Memory references

- `project_composer_default_guided_with_opt_out` — the design call this implements.
- `project_db_migration_policy` — informs Task 1's no-Alembic / delete-the-DB pattern.
- `feedback_no_calendar_shipping_commitments` — no calendar commitments in this plan.

---

## Review history

**2026-05-15 — Panel review (4 reviewers)**

Findings applied to this plan in the revision above:

- **Reality reviewer:** Confirmed `tests/integration/web/test_sessions_routes.py` and
  fixtures `client_with_user` / `client_anonymous` do not exist. The real fixture is
  `composer_test_client` in `conftest.py`, hardwired to `alice`. Task 4's test file now
  uses self-contained fixtures built from the conftest pattern.
- **Architecture reviewer:** Identified `_run_sync` as a reinvention of
  `run_sync_in_worker` (bypassing the single-worker pool and cancellation drain), the
  migration heuristic as dead code under the DB-delete policy, the `extra="forbid"`
  hedging as inconsistent with codebase practice, and the duplicated default-resolution
  logic (moot once the heuristic was retired).
- **Quality reviewer:** Identified accidental test isolation in the `service` fixture
  (multiple tests writing rows for the same `"alice"` user_id), missing route-level
  cross-user isolation test, missing DB-unavailability test, and the concurrent PATCH
  race window being accepted but unacknowledged.
- **Systems reviewer:** Identified the corrupt-mode PATCH lockout (a valid PATCH on a
  corrupt row would re-run the Tier-1 guard and crash even after the write succeeded),
  and noted the "Shifting the Burden" pattern where subsequent phases silently wipe
  Phase 1A user state under the DB-delete policy.

**2026-05-16 — Phase-validation panel re-review (cycle 1)**

Three findings closed in-place:

- **R1 / A1 — Pydantic config alignment.** ``ComposerPreferences`` was declared
  ``ConfigDict(frozen=True)`` and ``UpdateComposerPreferencesRequest`` was
  ``ConfigDict(frozen=True, extra="forbid")``. The codebase's web Pydantic models
  (``secrets/schemas.py``, ``blobs/schemas.py``, ``composer/progress.py``) use
  ``ConfigDict(strict=True, extra="forbid")`` and do not freeze. Both models in
  this plan are now aligned with that convention. ``strict=True`` reinforces the
  Tier-3 boundary (no implicit coercion); ``extra="forbid"`` rejects unknown keys
  on both request and response (the response is server-built, but the symmetry
  keeps the convention uniform across web models).
- **Q1 — Anonymous-auth fixture would 500 not 401.** Task 4's ``client_anonymous``
  fixture previously hedged: "leave get_current_user in place; if the default
  returns 401, no override is needed; if it 500s, add one." The real
  ``get_current_user`` (``auth/middleware.py:38–39``) reads
  ``app.state.auth_audit_recorder`` and ``app.state.settings`` *before* checking
  the Authorization header, so a bare ``FastAPI()`` would raise ``AttributeError``
  → 500 and the 401 tests (``test_get_requires_auth``, ``test_patch_requires_auth``)
  would fail as written. The fixture now installs an explicit
  ``dependency_overrides[get_current_user]`` that raises
  ``HTTPException(status_code=401)``, and ``raise_server_exceptions`` is left at
  the default (True) so any genuine 500 surfaces loudly.
