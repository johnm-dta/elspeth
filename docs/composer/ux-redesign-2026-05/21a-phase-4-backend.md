# Phase 4A — Backend: tutorial schema column + cache + run-path integration

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land the backend half of Phase 4 — extend `user_preferences_table`
with `tutorial_completed_at`, extend `PreferencesService` and the
`/api/composer-preferences` route to expose it, add a flat-file tutorial cache
keyed by `(canonical_prompt_sha256, model_id)`, and wire the cache into the
composer run path so that tutorial-mode runs of the canonical seed return
cached output deterministically.

**Architecture:** Schema-then-Pydantic-then-service-then-route-then-cache-
then-integration. Tests are TDD-shaped at every step. The new code paths reuse
all existing infrastructure (the Phase 1A `PreferencesService`, the existing
run-path, the existing Landscape readers); Phase 4A only adds the column, the
cache module, and a single call-site that consults the cache before invoking
the LLM.

**Tech Stack:** SQLAlchemy Core, FastAPI, Pydantic v2, pytest, hashlib (stdlib).

**Sibling plan:** [21b-phase-4-frontend.md](21b-phase-4-frontend.md) — frontend
tutorial container, six turn components, finalisation, skip, integration
test, smoke deploy.

**Overview document:** [21-phase-4-hello-world-tutorial.md](21-phase-4-hello-world-tutorial.md).

**Roadmap reference:** [00-implementation-roadmap.md](00-implementation-roadmap.md).

---

## Scope boundaries

**In scope:**

- Verify Phase 1A, 1B, 5a, 5b have shipped (Task 0 preflight).
- Add `tutorial_completed_at: timestamp NULL` to `user_preferences_table` on
  the shared metadata.
- Extend `ComposerPreferences` and `UpdateComposerPreferencesRequest` with
  the new field (Tier-3 boundary handling via Pydantic).
- Extend `PreferencesService`:
  - `_row_to_prefs` reads and Tier-1-guards the new field.
  - `get_composer_preferences` continues to return the full payload.
  - `update_composer_preferences` partial-update supports the new field;
    explicit `null` (i.e., "clear the tutorial-complete state") is **not**
    accepted via this PATCH — the Pydantic field default is `None` (meaning
    "leave alone"), and there is no client need to clear the flag (Phase 8
    handles "retake tutorial").
- Tutorial cache module:
  - `src/elspeth/web/preferences/tutorial_cache.py`.
  - SHA-256 keying on `(canonical_prompt, model_id)`.
  - Filesystem-backed (configurable directory; defaults to
    `${ELSPETH_DATA_DIR:-/var/lib/elspeth}/tutorial_cache/`).
  - Read returns `TutorialCacheEntry | None` (None = miss; entry = hit).
  - Write persists `TutorialCacheEntry` atomically (temp file + rename).
  - Tier-1 read-side guard: if a file exists but doesn't parse, crash.
- Integration test that proves the run path consults the cache when the user
  is in tutorial mode.
- Tier-1 read guard tests for both new code paths
  (`tutorial_completed_at` corruption, cache-file corruption).

**Out of scope:**

- A `tutorial_runs_table` or any per-attempt persistence. The cache is
  output-reuse; `tutorial_completed_at` is the completion gate. Nothing else.
- The frontend wiring (Phase 4B).
- A "force re-run live, don't cache" affordance. Operators delete the cache
  directory to invalidate.
- Telemetry of cache hit/miss rate. Phase 8 owns this.
- Alembic. The DB-delete pattern continues (per
  `project_db_migration_policy`).

## DB-delete cadence (Phase 9 owns the structural fix)

This is the **second** of three schema additions before Phase 9: (1) Phase 1A, (2) Phase 4A (`tutorial_completed_at`), (3) Phase 5b (`interpretation_events`). Each requires a DB-delete per `project_db_migration_policy`. A user who retakes the tutorial after Phase 4A's DB-delete is acceptable for staging. Structural fix (migration runner) is **owned by Phase 9**, post-launch. Cross-reference: roadmap §D5.

**Sequencing note:** If Phase 4A and Phase 5b deploy in the same operator-action window, batch both schema additions into one commit and perform a single DB-delete.

## New endpoints (Phase 4A additions)

Consumed by Phase 4B Part 2. Defined here so 21b2 can reference them.

**`POST /api/tutorial/run`** — body: `{"session_id": "<uuid>"}`. Response: `{"run_id": "<uuid>", "source_data_hash": "<hex>", "rows": [{"url":..., "score":..., "rationale":...}]}`. Cache-hit: milliseconds. Cache-miss: live run (~30s), populates cache on success. Non-tutorial-mode user → 400.

**`GET /api/sessions/{session_id}/runs/{run_id}/audit-story`** — response: `{"llm_call_count": N, "output_file_hash": "<hex>", "run_started_at": "<iso8601>", "plugin_versions": {...}}`. Reads from the Landscape (Tier 1). Not-found → 404. Landscape failure propagates — no fallback (design doc 04: "Otherwise the demonstration is theatre.").

## Trust tier check (per CLAUDE.md)

| Surface | Tier | Handling |
|---|---|---|
| Inbound `tutorial_completed_at` (PATCH body) | Tier 3 | Pydantic rejects non-datetime with 422. |
| Outbound `tutorial_completed_at` (DB read) | Tier 1 | `_row_to_prefs` guards: must be `None` or `datetime`; non-datetime → crash. |
| Tutorial cache file contents | Tier 1 | Parse failure = corruption → crash. |
| Tutorial cache file presence | n/a | Absent = miss, not fault. |
| Canonical seed prompt | Tier 1 | Python constant shared with frontend; drift → cache miss (intended). |
| LLM results in cache | Tier 1 once written | Cache write happens after Landscape record. Corruption → crash on parse. |

## File structure

**New:**

- `src/elspeth/web/preferences/tutorial_cache.py` — cache module.
- `tests/unit/web/preferences/test_tutorial_cache.py` — cache unit tests.
- `tests/integration/web/test_tutorial_cache_run_integration.py` — cache wiring.

**Modified:**

- `src/elspeth/web/sessions/models.py` — add `tutorial_completed_at` column.
- `src/elspeth/web/preferences/models.py` — extend Pydantic models.
- `src/elspeth/web/preferences/service.py` — extend read/write code paths.
- `tests/unit/web/preferences/test_schema.py` — extend expected-columns set.
- `tests/unit/web/preferences/test_models.py` — extend Pydantic tests.
- `tests/unit/web/preferences/test_service.py` — extend service tests.
- `tests/integration/web/test_preferences_routes.py` — extend route tests.
- The composer run-path file (identified during Task 7) — wire cache consult.

**Not modified:**

- `src/elspeth/web/preferences/routes.py` — Pydantic-model extension propagates
  automatically through `response_model`.

## Database migration note (operator action)

Task 1 requires a DB-delete before new code serves traffic. Phase 4B's smoke task performs it. If Phase 4A ships independently, operator must delete first. All users' `tutorial_completed_at` resets to NULL — every user retakes the tutorial on next login. See §"DB-delete cadence" for the full sequence context.

## Verification approach

Each task is TDD-shaped (failing test, run-to-fail, implement, run-to-pass,
commit). After Tasks 1–7 land, the Phase 4B integration tests and Playwright
smoke exercise the routes and the cache wiring end-to-end. The Phase 4B
smoke task performs the operator DB-delete and re-runs the full test suite.

---

## Task 0: Preflight — verify dependencies

**Files:** none modified.

This task is a **read-only verification step**. If any check fails, do not
proceed; surface the failure to the operator. Per CLAUDE.md "no scope
dumping", an agent that finds a gap here should fix it as part of this task
if it's tractable, or escalate explicitly to the operator if it isn't.

- [ ] **Step 1: Verify Phase 1A's table exists.**

Run:

```bash
.venv/bin/python -c "
from elspeth.web.sessions.models import metadata
table = metadata.tables['user_preferences']
columns = {c.name for c in table.columns}
expected = {'user_id', 'default_composer_mode', 'banner_dismissed_at', 'updated_at'}
assert columns == expected, f'unexpected columns: {columns}'
print('Phase 1A table OK')
"
```

Expected: `Phase 1A table OK`. Failure → halt; the Phase 1A table either does
not exist or has unexpected columns (perhaps a later phase already added
ours — read the file and decide).

- [ ] **Step 2: Verify Phase 1B's preferencesStore exists.**

Run:

```bash
test -f src/elspeth/web/frontend/src/stores/preferencesStore.ts && \
  grep -q "defaultMode" src/elspeth/web/frontend/src/stores/preferencesStore.ts && \
  echo "Phase 1B store OK"
```

Expected: `Phase 1B store OK`. Failure → halt.

- [ ] **Step 3: Verify Phase 5a's `inline_blob` source path.**

Run:

```bash
grep -l "inline_blob" src/elspeth/web/composer/*.py | head -5
```

Expected: at least one file (the composer skill or schema module) references
`inline_blob`. Failure → halt; Phase 5a not yet shipped.

- [ ] **Step 4: Verify Phase 5b's `interpretation_events_table`.**

Run:

```bash
.venv/bin/python -c "
from elspeth.web.sessions.models import metadata
assert 'interpretation_events' in metadata.tables, 'Phase 5b table missing'
t = metadata.tables['interpretation_events']
required = {'session_id', 'tool_call_id', 'user_term', 'llm_draft', 'accepted_value', 'resolved_at'}
present = {c.name for c in t.columns}
missing = required - present
assert not missing, f'Phase 5b interpretation_events missing columns: {missing}'
print('Phase 5b table OK')
"
```

Expected: `Phase 5b table OK`. Failure → halt.

- [ ] **Step 5: Verify Phase 5b records the prompt-template string.**

Read the `interpretation_events_table` definition in
`src/elspeth/web/sessions/models.py`. Confirm a column like
`accepted_value` exists and that the Phase 5b service writes the
prompt-template string into it on resolution. If the column shape differs
(e.g. it stores only an integer choice ID), halt and inform the operator —
the design doc 04 promise "recorded as a prompt template" cannot be met.

- [ ] **Step 6: No commit.** This task is verification only.

## Task 1: Schema — add `tutorial_completed_at` to `user_preferences_table`

**Files:**
- Modify: `src/elspeth/web/sessions/models.py`.
- Modify: `tests/unit/web/preferences/test_schema.py`.

- [ ] **Step 1: Write the failing test extension.**

Open `tests/unit/web/preferences/test_schema.py`. Update the columns-set
assertion to include the new column:

```python
def test_user_preferences_table_columns() -> None:
    """The user_preferences table has the expected columns."""
    table = metadata.tables["user_preferences"]
    column_names = {c.name for c in table.columns}
    assert column_names == {
        "user_id",
        "default_composer_mode",
        "banner_dismissed_at",
        "tutorial_completed_at",  # Phase 4
        "updated_at",
    }
```

Add a new test verifying the column is nullable and untyped-default-NULL:

```python
def test_tutorial_completed_at_is_nullable_with_no_default() -> None:
    """Tutorial completion is unset for new users; explicit completion bumps it.

    No server-side default — a row created via PATCH that does not set
    tutorial_completed_at must leave the column NULL. The Phase 4 finalisation
    flow always sets the timestamp explicitly.
    """
    table = metadata.tables["user_preferences"]
    column = table.c.tutorial_completed_at
    assert column.nullable is True
    assert column.server_default is None
```

- [ ] **Step 2: Run to fail.** `.venv/bin/python -m pytest tests/unit/web/preferences/test_schema.py -v` → FAIL (`tutorial_completed_at` not in columns set; `AttributeError` on `table.c.tutorial_completed_at`).

- [ ] **Step 3: Add the column.**

In `src/elspeth/web/sessions/models.py`, find the `user_preferences_table`
definition and add the new `Column` immediately before `updated_at`:

```python
user_preferences_table = Table(
    "user_preferences",
    metadata,
    Column("user_id", String, primary_key=True),
    Column(
        "default_composer_mode",
        String,
        nullable=False,
        server_default="guided",
    ),
    Column("banner_dismissed_at", DateTime(timezone=True), nullable=True),
    # Phase 4: tutorial-completion timestamp. NULL = tutorial not yet
    # completed (the user is in "tutorial mode" — the frontend renders the
    # hello-world tutorial container instead of the normal composer). Non-NULL
    # = tutorial has been completed (either via the standard 6-turn flow or
    # via the turn-1 skip affordance). The value is the timestamp at
    # finalisation; the timestamp itself is not load-bearing — only the
    # NULL / non-NULL distinction is read by the frontend.
    Column("tutorial_completed_at", DateTime(timezone=True), nullable=True),
    Column("updated_at", DateTime(timezone=True), nullable=False),
)
```

The existing `Column`, `String`, `DateTime` imports remain sufficient — no
new imports.

- [ ] **Step 4: Run test to verify it passes.**

```bash
.venv/bin/python -m pytest tests/unit/web/preferences/test_schema.py -v
```

Expected: PASS — all schema tests green.

- [ ] **Step 5: Commit.**

```bash
git add tests/unit/web/preferences/test_schema.py src/elspeth/web/sessions/models.py
git commit -m "feat(web): add tutorial_completed_at to user_preferences (Phase 4A.1)"
```

## Task 2: Pydantic models — add `tutorial_completed_at` field

**Files:**
- Modify: `src/elspeth/web/preferences/models.py`.
- Modify: `tests/unit/web/preferences/test_models.py`.

- [ ] **Step 1: Write the failing test extensions.**

Add to `tests/unit/web/preferences/test_models.py`:

```python
def test_composer_preferences_includes_tutorial_completed_at() -> None:
    """The full response payload now carries tutorial_completed_at."""
    payload = ComposerPreferences(
        default_mode="guided",
        banner_dismissed_at=None,
        tutorial_completed_at=None,
        updated_at=datetime.now(UTC),
    )
    assert payload.tutorial_completed_at is None


def test_composer_preferences_accepts_non_null_tutorial_completed_at() -> None:
    stamp = datetime(2026, 5, 15, 10, 0, tzinfo=UTC)
    payload = ComposerPreferences(
        default_mode="guided",
        banner_dismissed_at=None,
        tutorial_completed_at=stamp,
        updated_at=datetime.now(UTC),
    )
    assert payload.tutorial_completed_at == stamp


def test_update_request_accepts_tutorial_completed_at() -> None:
    stamp = datetime.now(UTC)
    payload = UpdateComposerPreferencesRequest(tutorial_completed_at=stamp)
    assert payload.tutorial_completed_at == stamp
    assert payload.default_mode is None
    assert payload.banner_dismissed_at is None


def test_update_request_rejects_non_datetime_tutorial_completed_at() -> None:
    """Tier-3 boundary: non-datetime values rejected with 422."""
    with pytest.raises(ValidationError):
        UpdateComposerPreferencesRequest(tutorial_completed_at="yesterday")  # type: ignore[arg-type]
```

- [ ] **Step 2: Run to fail.** `.venv/bin/python -m pytest tests/unit/web/preferences/test_models.py -v` → FAIL (`tutorial_completed_at` not a field on the Pydantic models).

- [ ] **Step 3: Extend the Pydantic models.**

In `src/elspeth/web/preferences/models.py`:

```python
class ComposerPreferences(BaseModel):
    """The full preferences payload returned by GET and PATCH."""

    model_config = ConfigDict(frozen=True)

    default_mode: ComposerMode
    banner_dismissed_at: datetime | None
    # Phase 4: NULL = user is in tutorial mode. Non-NULL = tutorial complete.
    tutorial_completed_at: datetime | None
    updated_at: datetime


class UpdateComposerPreferencesRequest(BaseModel):
    """Partial-update payload for PATCH."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    default_mode: ComposerMode | None = None
    banner_dismissed_at: datetime | None = None
    # Phase 4: the tutorial finalisation flow sets this to the completion
    # timestamp; no other client need exists. The PATCH semantics for None
    # are "leave alone" (the standard partial-update pattern). There is no
    # path to clear this to NULL via PATCH; operators who want to grant
    # a "retake" must SQL-UPDATE the row directly until Phase 8 ships the
    # settings affordance.
    tutorial_completed_at: datetime | None = None
```

- [ ] **Step 4: Run test to verify it passes.**

```bash
.venv/bin/python -m pytest tests/unit/web/preferences/test_models.py -v
```

Expected: PASS — all model tests green (existing + 4 new).

- [ ] **Step 5: Commit.**

```bash
git add src/elspeth/web/preferences/models.py tests/unit/web/preferences/test_models.py
git commit -m "feat(web): extend ComposerPreferences with tutorial_completed_at (Phase 4A.2)"
```

## Task 3: Service — extend read/write paths with Tier-1 read guard

**Files:**
- Modify: `src/elspeth/web/preferences/service.py`.
- Modify: `tests/unit/web/preferences/test_service.py`.

- [ ] **Step 1: Write the failing test extensions.**

Add to `tests/unit/web/preferences/test_service.py`:

```python
def test_get_for_new_user_has_null_tutorial_completed_at(service):
    """No-row users have no tutorial state."""
    prefs = asyncio.run(service.get_composer_preferences("alice-tutorial-new"))
    assert prefs.tutorial_completed_at is None


def test_patch_sets_tutorial_completed_at(service):
    stamp = datetime(2026, 5, 15, 11, 0, tzinfo=UTC)
    result = asyncio.run(
        service.update_composer_preferences(
            "alice-tutorial-set",
            UpdateComposerPreferencesRequest(tutorial_completed_at=stamp),
        )
    )
    assert result.tutorial_completed_at == stamp
    follow_up = asyncio.run(service.get_composer_preferences("alice-tutorial-set"))
    assert follow_up.tutorial_completed_at == stamp


def test_patch_can_set_mode_and_tutorial_in_one_call(service):
    """Tutorial finalisation: turn 6 PATCHes both fields atomically."""
    stamp = datetime(2026, 5, 15, 11, 5, tzinfo=UTC)
    result = asyncio.run(
        service.update_composer_preferences(
            "alice-tutorial-final",
            UpdateComposerPreferencesRequest(
                default_mode="freeform",
                tutorial_completed_at=stamp,
            ),
        )
    )
    assert result.default_mode == "freeform"
    assert result.tutorial_completed_at == stamp


def test_partial_update_preserves_tutorial_completed_at(service):
    """PATCHing only default_mode does not clear tutorial_completed_at."""
    stamp = datetime(2026, 5, 15, 11, 10, tzinfo=UTC)
    asyncio.run(
        service.update_composer_preferences(
            "alice-tutorial-preserve",
            UpdateComposerPreferencesRequest(tutorial_completed_at=stamp),
        )
    )
    # Subsequent PATCH that only changes the mode must not clear the tutorial flag.
    asyncio.run(
        service.update_composer_preferences(
            "alice-tutorial-preserve",
            UpdateComposerPreferencesRequest(default_mode="freeform"),
        )
    )
    follow_up = asyncio.run(service.get_composer_preferences("alice-tutorial-preserve"))
    assert follow_up.tutorial_completed_at == stamp
    assert follow_up.default_mode == "freeform"


def test_corrupt_tutorial_completed_at_crashes_loudly(service, engine):
    """Tier-1 guard: a stored value that's neither NULL nor a datetime crashes."""
    # Manually inject a value that SQLAlchemy's DateTime binding would normally
    # reject; we have to bypass SQLAlchemy to simulate corruption.
    with engine.begin() as conn:
        conn.execute(
            user_preferences_table.insert().values(
                user_id="alice-tutorial-corrupt",
                default_composer_mode="guided",
                banner_dismissed_at=None,
                tutorial_completed_at=None,
                updated_at=datetime.now(UTC),
            )
        )
        # Now corrupt the row via raw SQL (SQLite tolerates strings in a
        # DateTime-typed column; this models real-world corruption).
        conn.exec_driver_sql(
            "UPDATE user_preferences SET tutorial_completed_at = 'not-a-timestamp' "
            "WHERE user_id = 'alice-tutorial-corrupt'"
        )
    with pytest.raises(RuntimeError, match="tutorial_completed_at"):
        asyncio.run(service.get_composer_preferences("alice-tutorial-corrupt"))
```

- [ ] **Step 2: Run to fail.** `.venv/bin/python -m pytest tests/unit/web/preferences/test_service.py -v` → FAIL (service does not yet read or write `tutorial_completed_at`).

- [ ] **Step 3: Extend the service.**

In `src/elspeth/web/preferences/service.py`, update `_row_to_prefs` to read
and Tier-1-guard the new field:

```python
def _row_to_prefs(self, row: object, user_id: str) -> ComposerPreferences:
    """Convert a DB row to the response model with Tier-1 read guards."""
    mode = row.default_composer_mode  # type: ignore[union-attr]
    if mode not in _VALID_MODES:
        raise RuntimeError(
            f"user_preferences row for {user_id!r} has invalid "
            f"default_composer_mode={mode!r}"
        )
    tutorial_completed_at = row.tutorial_completed_at  # type: ignore[union-attr]
    # Tier-1 guard: must be None or a datetime. SQLite's tolerance for
    # type-violating values means a corrupt row may surface here as a string
    # or other type. Crash loudly with the offending value named so the
    # operator can diagnose.
    if tutorial_completed_at is not None and not isinstance(
        tutorial_completed_at, datetime
    ):
        raise RuntimeError(
            f"user_preferences row for {user_id!r} has invalid "
            f"tutorial_completed_at={tutorial_completed_at!r} "
            f"(expected datetime or None)"
        )
    return ComposerPreferences(
        default_mode=mode,
        banner_dismissed_at=row.banner_dismissed_at,  # type: ignore[union-attr]
        tutorial_completed_at=tutorial_completed_at,
        updated_at=row.updated_at,  # type: ignore[union-attr]
    )
```

Update the lazy-default branch of `get_composer_preferences` to include the
new field:

```python
return ComposerPreferences(
    default_mode=_DEFAULT_MODE,
    banner_dismissed_at=None,
    tutorial_completed_at=None,
    updated_at=self._now(),
)
```

Update `update_composer_preferences` to support the new partial-update field.
Mirror the existing pattern (read-resolve-preserve for fields not in the
payload):

```python
async def update_composer_preferences(
    self, user_id: str, payload: UpdateComposerPreferencesRequest
) -> ComposerPreferences:
    """Upsert the preferences row, touching only fields in payload.

    See Phase 1A.3 for the corrupt-mode PATCH lockout reasoning; the same
    pattern is extended here for tutorial_completed_at.

    # telemetry: deferred to Phase 8 polish — preference-change event.
    """
    now = self._now()

    def _sync() -> tuple[str, datetime | None, datetime | None]:
        """Returns (mode, banner_dismissed_at, tutorial_completed_at) after write."""
        with self._engine.begin() as conn:
            # Resolve mode (existing logic).
            if payload.default_mode is not None:
                insert_mode = payload.default_mode
            else:
                existing_mode = conn.execute(
                    select(user_preferences_table.c.default_composer_mode).where(
                        user_preferences_table.c.user_id == user_id
                    )
                ).scalar_one_or_none()
                insert_mode = existing_mode if existing_mode is not None else _DEFAULT_MODE

            # Resolve banner_dismissed_at (existing logic).
            if payload.banner_dismissed_at is not None:
                resolved_banner = payload.banner_dismissed_at
            else:
                resolved_banner = conn.execute(
                    select(user_preferences_table.c.banner_dismissed_at).where(
                        user_preferences_table.c.user_id == user_id
                    )
                ).scalar_one_or_none()

            # Resolve tutorial_completed_at (new — Phase 4).
            if payload.tutorial_completed_at is not None:
                resolved_tutorial = payload.tutorial_completed_at
            else:
                resolved_tutorial = conn.execute(
                    select(user_preferences_table.c.tutorial_completed_at).where(
                        user_preferences_table.c.user_id == user_id
                    )
                ).scalar_one_or_none()

            values: dict[str, object] = {
                "user_id": user_id,
                "default_composer_mode": insert_mode,
                "banner_dismissed_at": payload.banner_dismissed_at,
                "tutorial_completed_at": payload.tutorial_completed_at,
                "updated_at": now,
            }
            stmt = sqlite_insert(user_preferences_table).values(**values)
            update_clause: dict[str, object] = {"updated_at": now}
            if payload.default_mode is not None:
                update_clause["default_composer_mode"] = payload.default_mode
            if payload.banner_dismissed_at is not None:
                update_clause["banner_dismissed_at"] = payload.banner_dismissed_at
            if payload.tutorial_completed_at is not None:
                update_clause["tutorial_completed_at"] = payload.tutorial_completed_at
            stmt = stmt.on_conflict_do_update(
                index_elements=["user_id"], set_=update_clause
            )
            conn.execute(stmt)

        return insert_mode, resolved_banner, resolved_tutorial

    written_mode, written_banner, written_tutorial = await run_sync_in_worker(_sync)
    return ComposerPreferences(
        default_mode=written_mode,
        banner_dismissed_at=written_banner,
        tutorial_completed_at=written_tutorial,
        updated_at=now,
    )
```

Add the import for `datetime` at the top if not already present (it is in
Phase 1A's version).

- [ ] **Step 4: Run test to verify it passes.**

```bash
.venv/bin/python -m pytest tests/unit/web/preferences/test_service.py -v
```

Expected: PASS — all service tests green (existing + 5 new).

- [ ] **Step 5: Commit.**

```bash
git add src/elspeth/web/preferences/service.py tests/unit/web/preferences/test_service.py
git commit -m "feat(web): extend PreferencesService for tutorial_completed_at (Phase 4A.3)"
```

## Task 4: Route extension — verify the field round-trips

**Files:**
- Modify: `tests/integration/web/test_preferences_routes.py`.

No code change to `routes.py` — the Pydantic-model extension propagates
through `response_model` automatically. We add integration tests proving this.

- [ ] **Step 1: Write the failing test extensions.**

Add to `tests/integration/web/test_preferences_routes.py`:

```python
def test_get_returns_null_tutorial_for_new_user(client_as_alice: TestClient) -> None:
    response = client_as_alice.get("/api/composer-preferences")
    assert response.status_code == 200
    assert response.json()["tutorial_completed_at"] is None


def test_patch_sets_tutorial_completed_at(client_as_alice: TestClient) -> None:
    response = client_as_alice.patch(
        "/api/composer-preferences",
        json={"tutorial_completed_at": "2026-05-15T12:30:00Z"},
    )
    assert response.status_code == 200
    assert response.json()["tutorial_completed_at"] == "2026-05-15T12:30:00Z"


def test_patch_atomic_finalisation_payload(client_as_alice: TestClient) -> None:
    """Turn 6 finalisation: writes default_mode + tutorial_completed_at in one PATCH."""
    response = client_as_alice.patch(
        "/api/composer-preferences",
        json={
            "default_mode": "freeform",
            "tutorial_completed_at": "2026-05-15T12:35:00Z",
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["default_mode"] == "freeform"
    assert body["tutorial_completed_at"] == "2026-05-15T12:35:00Z"


def test_patch_rejects_non_datetime_tutorial(client_as_alice: TestClient) -> None:
    response = client_as_alice.patch(
        "/api/composer-preferences",
        json={"tutorial_completed_at": "yesterday"},
    )
    assert response.status_code == 422
```

- [ ] **Step 2: Run test to verify it fails.**

The new tests will run against the existing routes; if Tasks 1–3 are landed
correctly, they should already pass. Run:

```bash
.venv/bin/python -m pytest tests/integration/web/test_preferences_routes.py -v
```

Expected: PASS — all route tests green. If failures appear, diagnose
(typically a mismatch between the Pydantic model and the response payload).

- [ ] **Step 3: Commit.**

```bash
git add tests/integration/web/test_preferences_routes.py
git commit -m "test(web): cover tutorial_completed_at in preferences routes (Phase 4A.4)"
```

## Task 5: Tutorial cache module — flat-file storage

**Files:**
- Create: `src/elspeth/web/preferences/tutorial_cache.py`.
- Create: `tests/unit/web/preferences/test_tutorial_cache.py`.

The cache stores the **full run output** for a `(canonical_prompt, model)`
key. "Full run output" is whatever the run-path returns to the frontend on a
successful execution — pipeline state, run id, audit-trail snippet bundle,
result rows. We delegate the shape to the run path; the cache module sees an
opaque dict and serialises it.

**Cache directory:** `~/.elspeth_web/tutorial_cache/`. Operators can override
via a constructor argument (used by tests).

**File naming:** `<sha256_hex>.json` where the hex is the SHA-256 of
`f"{canonical_prompt}:{model_id}"`. The plain canonical prompt and model
are also stored inside the JSON for diagnostic visibility (an operator
inspecting a file should be able to confirm what it caches without
recomputing the hash).

**Tier-1 guarantees:**
- A file present → must parse via Pydantic. Parse failure → crash.
- A file absent → cache miss (legitimate).
- A file's recorded `(canonical_prompt, model_id)` must match what we expected
  for the given key — if not, the file is in the wrong location and we crash.
  (This guards against a misconfigured operator copying files around.)

- [ ] **Step 1: Write the failing test.**

Create `tests/unit/web/preferences/test_tutorial_cache.py`:

```python
"""Tests for the tutorial-run flat-file cache."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from elspeth.web.preferences.tutorial_cache import (
    CANONICAL_SEED_PROMPT,
    TutorialCache,
    TutorialCacheEntry,
)


@pytest.fixture
def cache_dir(tmp_path: Path) -> Path:
    d = tmp_path / "tutorial_cache"
    d.mkdir()
    return d


@pytest.fixture
def cache(cache_dir: Path) -> TutorialCache:
    return TutorialCache(directory=cache_dir)


def test_canonical_seed_prompt_constant_is_exact() -> None:
    """The seed prompt must match design doc 04 verbatim."""
    assert CANONICAL_SEED_PROMPT == (
        "create a list of 5 government web pages and use an LLM to rate "
        "how cool they are"
    )


def test_lookup_returns_none_on_miss(cache: TutorialCache) -> None:
    assert cache.lookup(CANONICAL_SEED_PROMPT, "claude-opus-4-7") is None


def test_lookup_returns_entry_on_hit(cache: TutorialCache) -> None:
    entry = TutorialCacheEntry(
        canonical_prompt=CANONICAL_SEED_PROMPT,
        model_id="claude-opus-4-7",
        cached_at=datetime(2026, 5, 15, tzinfo=UTC),
        run_output={"run_id": "abc-123", "rows": [{"url": "ato.gov.au"}]},
        source_data_hash="a7f3e2deadbeef",
        interpretation_event_id="evt-1",
    )
    cache.store(entry)
    got = cache.lookup(CANONICAL_SEED_PROMPT, "claude-opus-4-7")
    assert got is not None
    assert got.canonical_prompt == CANONICAL_SEED_PROMPT
    assert got.model_id == "claude-opus-4-7"
    assert got.source_data_hash == "a7f3e2deadbeef"
    assert got.run_output["run_id"] == "abc-123"


def test_lookup_misses_on_different_model(cache: TutorialCache) -> None:
    entry = TutorialCacheEntry(
        canonical_prompt=CANONICAL_SEED_PROMPT,
        model_id="claude-opus-4-7",
        cached_at=datetime(2026, 5, 15, tzinfo=UTC),
        run_output={},
        source_data_hash="hash",
        interpretation_event_id=None,
    )
    cache.store(entry)
    assert cache.lookup(CANONICAL_SEED_PROMPT, "claude-opus-4-8") is None


def test_lookup_misses_on_different_prompt(cache: TutorialCache) -> None:
    entry = TutorialCacheEntry(
        canonical_prompt=CANONICAL_SEED_PROMPT,
        model_id="claude-opus-4-7",
        cached_at=datetime(2026, 5, 15, tzinfo=UTC),
        run_output={},
        source_data_hash="hash",
        interpretation_event_id=None,
    )
    cache.store(entry)
    edited = CANONICAL_SEED_PROMPT + " and also rate accessibility"
    assert cache.lookup(edited, "claude-opus-4-7") is None


def test_store_and_lookup_round_trip(cache: TutorialCache, cache_dir: Path) -> None:
    entry = TutorialCacheEntry(
        canonical_prompt=CANONICAL_SEED_PROMPT,
        model_id="claude-opus-4-7",
        cached_at=datetime(2026, 5, 15, tzinfo=UTC),
        run_output={"k": "v"},
        source_data_hash="hash",
        interpretation_event_id="evt-1",
    )
    cache.store(entry)
    files = list(cache_dir.iterdir())
    assert len(files) == 1
    assert files[0].suffix == ".json"
    # The file should be readable as JSON; the prompt and model are visible
    # in the JSON for diagnostic-by-cat purposes.
    raw = json.loads(files[0].read_text())
    assert raw["canonical_prompt"] == CANONICAL_SEED_PROMPT
    assert raw["model_id"] == "claude-opus-4-7"


def test_corrupt_file_crashes_lookup(cache: TutorialCache, cache_dir: Path) -> None:
    """Tier-1 guard: a present-but-unparseable file is a fault, not a miss."""
    # Compute the key the way the cache does (this mirrors the internal hash).
    from elspeth.web.preferences.tutorial_cache import _compute_key
    key = _compute_key(CANONICAL_SEED_PROMPT, "claude-opus-4-7")
    (cache_dir / f"{key}.json").write_text("this is not json")
    with pytest.raises(RuntimeError, match="tutorial cache"):
        cache.lookup(CANONICAL_SEED_PROMPT, "claude-opus-4-7")


def test_file_with_mismatched_prompt_crashes_lookup(
    cache: TutorialCache, cache_dir: Path
) -> None:
    """Tier-1 guard: an in-place file whose contents disagree with the key.

    Models a misconfigured operator who copied a file to the wrong name.
    """
    from elspeth.web.preferences.tutorial_cache import _compute_key
    key = _compute_key(CANONICAL_SEED_PROMPT, "claude-opus-4-7")
    bad_entry = {
        "canonical_prompt": "a different prompt",
        "model_id": "claude-opus-4-7",
        "cached_at": "2026-05-15T00:00:00+00:00",
        "run_output": {},
        "source_data_hash": "hash",
        "interpretation_event_id": None,
    }
    (cache_dir / f"{key}.json").write_text(json.dumps(bad_entry))
    with pytest.raises(RuntimeError, match="prompt mismatch"):
        cache.lookup(CANONICAL_SEED_PROMPT, "claude-opus-4-7")


def test_store_is_atomic(cache: TutorialCache, cache_dir: Path) -> None:
    """Write goes through a tempfile + rename; a half-written file is impossible."""
    entry = TutorialCacheEntry(
        canonical_prompt=CANONICAL_SEED_PROMPT,
        model_id="claude-opus-4-7",
        cached_at=datetime(2026, 5, 15, tzinfo=UTC),
        run_output={"k": "v"},
        source_data_hash="hash",
        interpretation_event_id=None,
    )
    cache.store(entry)
    # The directory should contain exactly one .json file and no tempfiles.
    files = list(cache_dir.iterdir())
    assert len(files) == 1
    assert files[0].suffix == ".json"
```

- [ ] **Step 2: Run to fail.** `.venv/bin/python -m pytest tests/unit/web/preferences/test_tutorial_cache.py -v` → FAIL (`ModuleNotFoundError: tutorial_cache`).

- [ ] **Step 3: Implement the cache module.**

Create `src/elspeth/web/preferences/tutorial_cache.py`:

```python
"""Flat-file tutorial-seed run cache. Absence = miss; corruption = crash.
Key: SHA-256(f"{canonical_prompt}:{model_id}"). Invalidate by deleting directory."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, ValidationError

# The exact prompt from design doc 04 §"The canonical seed prompt".
# Backend and frontend share this constant; if they drift, the cache will
# miss for the canonical click-through and operators will see live LLM calls
# during the tutorial. The drift is itself the signal.
CANONICAL_SEED_PROMPT = (
    "create a list of 5 government web pages and use an LLM to rate "
    "how cool they are"
)


def _default_cache_dir() -> Path:
    """Default cache dir from ELSPETH_DATA_DIR env var (falls back to /var/lib/elspeth)."""
    data_dir = os.environ.get("ELSPETH_DATA_DIR", "/var/lib/elspeth")
    return Path(data_dir) / "tutorial_cache"


def _compute_key(canonical_prompt: str, model_id: str) -> str:
    """Hex SHA-256 of ``f"{canonical_prompt}:{model_id}"``."""
    h = hashlib.sha256()
    h.update(canonical_prompt.encode("utf-8"))
    h.update(b":")
    h.update(model_id.encode("utf-8"))
    return h.hexdigest()


class TutorialCacheEntry(BaseModel):
    """Cached canonical-seed run output. `run_output` is opaque (frontend knows shape)."""

    model_config = ConfigDict(frozen=True)

    canonical_prompt: str
    model_id: str
    cached_at: datetime
    run_output: dict[str, Any]
    source_data_hash: str  # design doc 04 turn 5
    interpretation_event_id: str | None  # Phase 5b row; None if not recorded


class TutorialCache:
    """Flat-file cache for canonical-seed run outputs."""

    def __init__(self, *, directory: Path | None = None) -> None:
        self._dir = directory if directory is not None else _default_cache_dir()

    def lookup(self, canonical_prompt: str, model_id: str) -> TutorialCacheEntry | None:
        """Return cached entry, or None on miss. Crashes on corruption."""
        key = _compute_key(canonical_prompt, model_id)
        path = self._dir / f"{key}.json"
        if not path.exists():
            return None
        try:
            raw = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise RuntimeError(f"tutorial cache file unreadable: {path}") from exc
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"tutorial cache file is not valid JSON: {path}") from exc
        try:
            entry = TutorialCacheEntry.model_validate(data)
        except ValidationError as exc:
            raise RuntimeError(
                f"tutorial cache file does not match expected shape: {path}"
            ) from exc
        if entry.canonical_prompt != canonical_prompt or entry.model_id != model_id:
            raise RuntimeError(
                f"tutorial cache file {path} prompt mismatch: "
                f"file recorded ({entry.canonical_prompt!r}, {entry.model_id!r}) "
                f"but lookup was for ({canonical_prompt!r}, {model_id!r})"
            )
        return entry

    def store(self, entry: TutorialCacheEntry) -> None:
        """Persist the entry atomically (tempfile + os.replace)."""
        self._dir.mkdir(parents=True, exist_ok=True)
        key = _compute_key(entry.canonical_prompt, entry.model_id)
        final_path = self._dir / f"{key}.json"
        fd, tmp_path_str = tempfile.mkstemp(
            prefix=f"{key}.", suffix=".json.tmp", dir=str(self._dir)
        )
        tmp_path = Path(tmp_path_str)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(entry.model_dump_json())
            os.replace(tmp_path, final_path)
        except Exception:
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass
            raise
```

- [ ] **Step 4: Run test to verify it passes.**

```bash
.venv/bin/python -m pytest tests/unit/web/preferences/test_tutorial_cache.py -v
```

Expected: PASS — all cache tests green.

- [ ] **Step 4b: Plan — corrupt-cache propagation integration test (implemented in Task 7).**

Task 7 creates `tests/integration/web/test_tutorial_cache_run_integration.py`
and has access to the run-path fixtures. The corrupt-cache propagation test is
added there rather than here (this module has no run-path fixtures yet). See
Task 7 Step 2 for the `test_corrupt_cache_file_crashes_run_path` test body.

- [ ] **Step 5: Commit.**

```bash
git add src/elspeth/web/preferences/tutorial_cache.py tests/unit/web/preferences/test_tutorial_cache.py
git commit -m "feat(web): add flat-file tutorial-seed run cache (Phase 4A.5)"
```

## Task 6: Wire the cache into the FastAPI app composition site

**Files:**
- Modify: the FastAPI app-composition site (same file as Phase 1A Task 4).

The cache is constructed once per app, attached to `app.state` alongside
`preferences_service`. The run path (Task 7) reads from `app.state.tutorial_cache`.

- [ ] **Step 1: Identify the FastAPI app-composition site.**

```bash
grep -rn "create_preferences_router\b" src/elspeth/web --include="*.py" | grep -v __pycache__ | grep -v test
```

Read the hit file end-to-end. Locate the line that wires
`app.state.preferences_service`.

- [ ] **Step 2: Add cache construction.**

Add to the app-composition site, immediately after the preferences_service line:

```python
from elspeth.web.preferences.tutorial_cache import TutorialCache

# Phase 4: tutorial-run cache.
# Directory = ${ELSPETH_DATA_DIR:-/var/lib/elspeth}/tutorial_cache/
# Add tutorial_cache_dir to WebSettings and pass it here; the startup
# sequence must verify the service user has write permission on that
# directory and raise RuntimeError at startup if not.
# Operators invalidate by deleting the directory.
app.state.tutorial_cache = TutorialCache(
    directory=settings.tutorial_cache_dir
)
```

Add `tutorial_cache_dir: Path` to `WebSettings` with a validator that
resolves `ELSPETH_DATA_DIR` and asserts write permission. If write is not
available, raise `RuntimeError` at startup with a clear message naming the
path and the required OS user.

- [ ] **Step 3: Smoke-test that the app still starts.**

```bash
.venv/bin/python -m pytest tests/integration/web/ -v -x
```

Expected: PASS — no existing integration tests should break.

- [ ] **Step 4: Commit.**

```bash
git add <app-file>
git commit -m "feat(web): wire TutorialCache onto app.state (Phase 4A.6)"
```

## Task 7: Run-path integration — cache consult under tutorial mode

**Files:**
- Modify: the composer run-path file (identified during recon below).
- Create: `tests/integration/web/test_tutorial_cache_run_integration.py`.

This is the most architecturally delicate task in the plan. The composer
run path is a single call site (e.g., the `POST /api/sessions/{id}/runs`
handler or a service method invoked from it); we add a single early-return
branch that:

1. Checks `request.app.state.preferences_service.get_composer_preferences(
   user.user_id).tutorial_completed_at is None` (the user is in tutorial mode).
2. Checks whether the about-to-run pipeline matches the canonical-seed
   shape (the source is an `inline_blob` whose content equals or hashes
   to the canonical seed; the transforms are `web_scrape` + `llm_rate` in
   that order; the sink is the standard tutorial-sink shape — confirmed
   during recon).
3. Calls `app.state.tutorial_cache.lookup(CANONICAL_SEED_PROMPT, model_id)`.
4. On hit: returns the cached output (synthesised into the run-response
   shape the frontend expects).
5. On miss: proceeds with the normal run path. **After** the run completes
   successfully, calls `app.state.tutorial_cache.store(...)` to populate the
   cache for the next user.

**Edge: a user edits the seed.** The canonical-seed-match check fails; the
cache is bypassed entirely; the user pays for a live LLM run. This is the
intended behaviour — caching edited prompts would require a much larger key
space and create per-edit cache churn.

- [ ] **Step 1: Reconnaissance — identify the run path.**

```bash
grep -rn "session_engine\|sessions/runs\|run_pipeline" src/elspeth/web --include="*.py" \
  | grep -v __pycache__ | grep -v test | head -30
```

Read the file containing the actual run-execution entry point. Identify:

1. The function that the route handler calls to execute a pipeline.
2. How the function receives the model id (or how to derive it from the
   pipeline config).
3. The shape of the return value (so we can synthesise the cache-hit
   response in the same shape).
4. The point at which "the pipeline has just finished successfully" is known
   (so we can call `cache.store` there).

Write down (in a scratch note or commit message body) the names and types
you found. Do not guess.

- [ ] **Step 2: Write the failing integration test.**

Create `tests/integration/web/test_tutorial_cache_run_integration.py`:

```python
"""Integration test: tutorial-mode runs of the canonical seed consult the cache."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.pool import StaticPool

from elspeth.web.auth.middleware import get_current_user
from elspeth.web.auth.models import UserIdentity
from elspeth.web.preferences.routes import create_preferences_router
from elspeth.web.preferences.service import PreferencesService
from elspeth.web.preferences.tutorial_cache import (
    CANONICAL_SEED_PROMPT,
    TutorialCache,
    TutorialCacheEntry,
)
from elspeth.web.sessions.engine import create_session_engine
from elspeth.web.sessions.schema import initialize_session_schema


@pytest.fixture
def cache_dir(tmp_path: Path) -> Path:
    d = tmp_path / "cache"
    d.mkdir()
    return d


@pytest.fixture
def app_with_cache(cache_dir: Path) -> FastAPI:
    engine = create_session_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    initialize_session_schema(engine)
    app = FastAPI()
    app.state.session_engine = engine
    app.state.preferences_service = PreferencesService(engine)
    app.state.tutorial_cache = TutorialCache(directory=cache_dir)

    identity = UserIdentity(user_id="alice", username="alice")

    async def _mock_user() -> UserIdentity:
        return identity

    app.dependency_overrides[get_current_user] = _mock_user
    app.include_router(create_preferences_router())
    # The actual run-path router is wired in during the implementation
    # step — this test exercises the live wiring against a real session
    # creation + run.
    from <run-path-router-module> import create_run_router  # noqa
    app.include_router(create_run_router())
    return app


def test_cache_hit_returns_cached_output_without_calling_llm(
    app_with_cache: FastAPI, cache_dir: Path
) -> None:
    """User in tutorial mode + canonical seed + cache hit → no LLM call."""
    # Pre-populate the cache.
    entry = TutorialCacheEntry(
        canonical_prompt=CANONICAL_SEED_PROMPT,
        model_id="claude-opus-4-7",
        cached_at=datetime(2026, 5, 15, tzinfo=UTC),
        run_output={"rows": [{"url": "ato.gov.au", "score": 5}]},
        source_data_hash="a7f3e2cached",
        interpretation_event_id="evt-cached",
    )
    app_with_cache.state.tutorial_cache.store(entry)

    # The user is in tutorial mode (no PATCH yet → tutorial_completed_at is None).
    client = TestClient(app_with_cache)

    # Patch the LLM-call path so we can assert it was NOT called.
    with patch("<llm-call-path>") as mock_llm:
        # Build the canonical-seed pipeline through the standard composer
        # API (the test isn't unit-testing the cache module — it's verifying
        # the integration). Specifics depend on the run-path shape, filled
        # in during implementation.
        response = client.post(
            "/api/sessions/<session_id>/runs",
            json={"<canonical-seed-pipeline-shape>": "..."},
        )
        assert response.status_code == 200
        body = response.json()
        # The cached run_output is returned verbatim.
        assert body["rows"][0]["url"] == "ato.gov.au"
        # The cached source_data_hash propagates to the run response so
        # turn 5 can render it without a separate Landscape read.
        assert body["source_data_hash"] == "a7f3e2cached"
        # The LLM was never called.
        mock_llm.assert_not_called()


def test_cache_miss_runs_live_then_populates_cache(
    app_with_cache: FastAPI, cache_dir: Path
) -> None:
    """Tutorial mode + canonical seed + empty cache → live run + cache write."""
    client = TestClient(app_with_cache)
    # No pre-populate; cache is empty.
    response = client.post(
        "/api/sessions/<session_id>/runs",
        json={"<canonical-seed-pipeline-shape>": "..."},
    )
    assert response.status_code == 200
    # After the run, the cache should contain one entry.
    files = list(cache_dir.iterdir())
    assert len(files) == 1


def test_non_tutorial_user_skips_cache(
    app_with_cache: FastAPI, cache_dir: Path
) -> None:
    """User with tutorial_completed_at set → cache is bypassed entirely."""
    # Mark Alice's tutorial complete.
    client = TestClient(app_with_cache)
    client.patch(
        "/api/composer-preferences",
        json={"tutorial_completed_at": "2026-05-14T00:00:00Z"},
    )
    # Pre-populate the cache with a recognisable entry.
    entry = TutorialCacheEntry(
        canonical_prompt=CANONICAL_SEED_PROMPT,
        model_id="claude-opus-4-7",
        cached_at=datetime(2026, 5, 15, tzinfo=UTC),
        run_output={"rows": [{"url": "cached-url", "score": 99}]},
        source_data_hash="should-not-appear",
        interpretation_event_id=None,
    )
    app_with_cache.state.tutorial_cache.store(entry)

    response = client.post(
        "/api/sessions/<session_id>/runs",
        json={"<canonical-seed-pipeline-shape>": "..."},
    )
    assert response.status_code == 200
    # The cache hit value MUST NOT appear in the response — the user is
    # past the tutorial; their run is live.
    body = response.json()
    assert body.get("source_data_hash") != "should-not-appear"


def test_edited_prompt_skips_cache(
    app_with_cache: FastAPI, cache_dir: Path
) -> None:
    """Tutorial mode + edited prompt → cache is bypassed; live run."""
    # Pre-populate the cache for the canonical seed.
    entry = TutorialCacheEntry(
        canonical_prompt=CANONICAL_SEED_PROMPT,
        model_id="claude-opus-4-7",
        cached_at=datetime(2026, 5, 15, tzinfo=UTC),
        run_output={"rows": []},
        source_data_hash="cached-hash",
        interpretation_event_id=None,
    )
    app_with_cache.state.tutorial_cache.store(entry)

    client = TestClient(app_with_cache)
    # Build a pipeline from an edited prompt (canonical + extra clause).
    response = client.post(
        "/api/sessions/<session_id>/runs",
        json={"<edited-pipeline-shape>": "..."},
    )
    assert response.status_code == 200
    body = response.json()
    # Live run; the cached source_data_hash must not appear.
    assert body.get("source_data_hash") != "cached-hash"


def test_corrupt_cache_file_crashes_run_path(
    app_with_cache: FastAPI, cache_dir: Path
) -> None:
    """Tier-1 guarantee: corrupt cache must crash (500), not silently bypass.

    Guards against a defensive try/except being added around cache.lookup.
    """
    from elspeth.web.preferences.tutorial_cache import _compute_key
    key = _compute_key(CANONICAL_SEED_PROMPT, "claude-opus-4-7")
    (cache_dir / f"{key}.json").write_text("not valid json {{{")
    client = TestClient(app_with_cache)
    response = client.post(
        "/api/sessions/<session_id>/runs",
        json={"<canonical-seed-pipeline-shape>": "..."},
    )
    assert response.status_code == 500
```

The `<placeholder>` strings are filled in from Step 1's recon. The test
file may not be valid Python until then — intentional: TDD must fail
against the real run path, not against a mock.

- [ ] **Step 3: Run test to verify it fails.**

```bash
.venv/bin/python -m pytest tests/integration/web/test_tutorial_cache_run_integration.py -v
```

Expected: FAIL — either an import error (placeholders unresolved) or a
behavioural failure (cache not yet consulted).

- [ ] **Step 4: Implement the cache-consult hook in the run path.**

In the run-path file identified during recon, add a guard at the very top of
the execution entry point:

```python
from elspeth.web.preferences.tutorial_cache import CANONICAL_SEED_PROMPT

async def execute_pipeline_run(
    request: Request,
    user: UserIdentity,
    session_id: str,
    pipeline_state: dict,
) -> dict:
    """Execute a pipeline run for the user.

    Phase 4: tutorial-mode users running the canonical seed pipeline consult
    the cache first. On hit, return the cached output (synthesised into the
    run-response shape). On miss, run live; on success, populate the cache.
    """
    # Tutorial-mode + canonical-seed cache check.
    prefs = await request.app.state.preferences_service.get_composer_preferences(
        user.user_id
    )
    if prefs.tutorial_completed_at is None:
        # User is in tutorial mode. Check if the pipeline is the canonical seed.
        if _is_canonical_seed_pipeline(pipeline_state):
            model_id = _model_id_for_pipeline(pipeline_state)
            cache_entry = request.app.state.tutorial_cache.lookup(
                CANONICAL_SEED_PROMPT, model_id
            )
            if cache_entry is not None:
                # Cache hit — return the cached output in the run-response shape.
                return _cached_entry_to_run_response(cache_entry)

    # Normal path: run live.
    result = await _execute_pipeline_live(request, user, session_id, pipeline_state)

    # Tutorial-mode + canonical-seed cache populate (on success only).
    if (
        prefs.tutorial_completed_at is None
        and _is_canonical_seed_pipeline(pipeline_state)
        and _is_successful_run(result)
    ):
        model_id = _model_id_for_pipeline(pipeline_state)
        request.app.state.tutorial_cache.store(
            TutorialCacheEntry(
                canonical_prompt=CANONICAL_SEED_PROMPT,
                model_id=model_id,
                cached_at=datetime.now(UTC),
                run_output=result["run_output"],
                source_data_hash=result["source_data_hash"],
                interpretation_event_id=result.get("interpretation_event_id"),
            )
        )

    return result
```

The `_is_canonical_seed_pipeline`, `_model_id_for_pipeline`,
`_cached_entry_to_run_response`, `_is_successful_run`, and
`_execute_pipeline_live` helpers are defined in the same file. Their shape
depends on recon; the executor fills them in based on the actual run-path
internals.

**Tier discipline:** cache `lookup`/`store` are Tier-1 (failures crash);
`_is_canonical_seed_pipeline` reads Tier-2 `pipeline_state` (no-match is
normal, corrupt structure crashes); `model_id` extraction crashes on
absence (it's required by every live run too).

- [ ] **Step 5: Run test to verify it passes.**

```bash
.venv/bin/python -m pytest tests/integration/web/test_tutorial_cache_run_integration.py -v
```

Expected: PASS — all integration tests green.

- [ ] **Step 6: Run the full integration suite to catch regressions.**

```bash
.venv/bin/python -m pytest tests/integration/web/ -v
```

Expected: PASS — no existing run-path tests break. If they do, the
cache-consult branch is firing for non-tutorial users (check the
`prefs.tutorial_completed_at is None` gate); fix and re-run.

- [ ] **Step 7: Commit.**

```bash
git add <run-path-file> tests/integration/web/test_tutorial_cache_run_integration.py
git commit -m "feat(web): consult tutorial cache on canonical-seed runs (Phase 4A.7)"
```

---

## What Phase 4A leaves the backend in

After Tasks 0–7: new column, Tier-1 guards, cache module (filesystem-backed,
corruption-detecting), and run-path cache consult. Phase 4B wires the frontend.

## Risks and mitigations

Key risks: run-path entry point not where assumed (Task 7 Step 1 recon resolves); cache fires for non-tutorial users (gate on `tutorial_completed_at is None` + regression test); model_id derivation drifts (integration tests assert hit-on-pre-populated-cache); concurrent writers (atomic via `os.replace`).

## Forward compatibility

Phase 8 schema additions wipe `tutorial_completed_at` via DB-delete policy — every user retakes the tutorial. Structural fix (Alembic) owned by the roadmap.

## Memory references

- `project_composer_first_run_tutorial`
- `project_composer_canonical_test_case`
- `project_composer_dynamic_source_from_chat`
- `project_composer_default_guided_with_opt_out`
- `project_db_migration_policy`
- `feedback_no_calendar_shipping_commitments`

---

## Review history

### 2026-05-15 — review panel

| ID | Severity | Status | Summary |
|---|---|---|---|
| 4A-F1 | BLOCKER (Systems) | Applied | DB-delete cadence section added after §Scope boundaries |
| 4A-F2 | CRITICAL (Systems) | Applied | Cache path changed to `ELSPETH_DATA_DIR`; `WebSettings` field added; startup permission check |
| 4A-F3 | CRITICAL (Quality) | Applied | Corrupt-cache integration test added to Task 5; orchestrator-propagation integration test added |
| 4A-F4 | IMPORTANT (Architecture) | Applied | Sequencing note added to §DB-delete cadence |
| 4A-F5 | IMPORTANT (Quality) | Applied | Preflight Task 0 Step 4 column check made precise |
| 4A-F6 (from 4B2-F1) | IMPORTANT (Architecture) | Applied | `POST /api/tutorial/run` route specified in §New endpoints |
| 4A-F7 (from 4B2-F3) | IMPORTANT (Quality) | Applied | `GET /api/sessions/{id}/runs/{run_id}/audit-story` specified in §New endpoints |
