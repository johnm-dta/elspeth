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
  - `update_composer_preferences` partial-update supports the new field with
    **three semantic states**: (a) field absent from the PATCH body → no-op
    for the column; (b) field present with a `datetime` value → write the
    timestamp; (c) field present with explicit `null` → write `NULL` (the
    Phase 8 retake path). Absent-vs-`null` is distinguished via Pydantic v2's
    `model_fields_set` so the service does not collapse the two cases. See
    §"Cross-plan contract — `tutorial_completed_at` PATCH semantics" below.
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

## Cross-plan contract — `tutorial_completed_at` PATCH semantics

Phase 4 ships `tutorial_completed_at: datetime | None` to
`user_preferences_table`. The field is nullable in the schema AND on the
PATCH request body. The three semantic states are:

| Field state in PATCH body | Service behaviour | Used by |
|---|---|---|
| Absent (key not in payload) | No-op for this column | Most Phase 4 PATCH flows (e.g., mode toggle, banner-dismiss) |
| Present with `datetime` value | Write timestamp to column | Phase 4 turn 6 finalisation; Phase 4 turn 1 skip |
| Present with `null` | Write `NULL` to column | Phase 8 retake (`20-phase-8-polish-and-telemetry.md` Task 6) |

This contract is **co-owned** by Phase 4 (which ships the schema, the
Pydantic models, the service, and the Tier-1 read-side guard) and Phase 8
(which adds the retake mechanism by PATCHing `tutorial_completed_at: null`).
Either side changing the nullification semantics requires a paired edit in
both plan files.

Audit primacy: a successful PATCH that nulls the column (the retake event)
is a user-write-intent that belongs in the Landscape — see Phase 8 Task 6
for the audit-emit boundary question and the prior-timestamp capture
requirement. Phase 4's service emits the existing
`composer.preferences.patch_total` counter unchanged; the new
audit-event-on-retake responsibility lives with Phase 8.

The frontend exposes a derived boolean `tutorialCompleted = (tutorialCompletedAt !== null)` on the Zustand `preferencesStore`. The retake path does **not** add a new store action or a new API client function — it reuses `api.updateComposerPreferences({ tutorial_completed_at: null })`.

## New endpoints (Phase 4A additions)

Consumed by Phase 4B Part 2. Defined here so 21b2 can reference them.

**`POST /api/tutorial/run`** — body: `{"session_id": "<uuid>"}`. Response: `{"run_id": "<uuid>", "source_data_hash": "<hex>", "rows": [{"url":..., "score":..., "rationale":...}]}`. Cache-hit: the backend **synthesises a real Landscape entry under the current user's session**, populated from the cached content (rows, source_data_hash, llm_call_count=0, pipeline_yaml) plus a `seeded_from_cache: true` marker carrying the cache key. The returned `run_id` is **owned by the current session** — there is no foreign-run reference in the response. Cache-miss: live run (~30s), populates cache on success. Non-tutorial-mode user → 400.

**`GET /api/sessions/{session_id}/runs/{run_id}/audit-story`** — response: `{"llm_call_count": N, "output_file_hash": "<hex>", "run_started_at": "<iso8601>", "plugin_versions": {...}, "seeded_from_cache": <bool>, "cache_key": "<hex>" | null}`. Reads from the Landscape — server-generated content (operational Tier-1 behaviour: corruption crashes, absence is 404). When the run was a cache hit, `seeded_from_cache` is `true`, `llm_call_count` is `0`, and `cache_key` is the SHA-256 that points at the original cache-seeding run for cross-run lineage joins. Not-found → 404. Landscape failure propagates — no fallback (design doc 04: "Otherwise the demonstration is theatre.").

## Trust tier check (per CLAUDE.md)

| Surface | Tier | Handling |
|---|---|---|
| Inbound `tutorial_completed_at` (PATCH body) | Tier 3 | Pydantic rejects non-datetime with 422. |
| Outbound `tutorial_completed_at` (DB read) | Tier 1 | `_row_to_prefs` guards: must be `None` or `datetime`; non-datetime → crash. |
| Tutorial cache file contents | server-generated cache content | Parse failure = corruption → crash. (Final tier classification deferred to P23; the operational behaviour — crash-on-corrupt, miss-on-absence — is the binding contract.) |
| Tutorial cache file presence | n/a | Absent = miss, not fault. |
| Canonical seed prompt | constant | Python constant shared with frontend; drift → cache miss (intended). |
| LLM results in cache | server-generated cache content | Cache write happens after the canonical-seed run is recorded in the Landscape and only when every row succeeded (gating is added by P18; see Task 7 Step 4's `# TODO: P18` hook). Corruption → crash on parse. |

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
    #
    # `nullable=True` is load-bearing for the Phase 8 retake mechanism: the
    # Phase 8 retake button PATCHes `{tutorial_completed_at: null}` to clear
    # the column and re-enter tutorial mode (see §"Cross-plan contract —
    # `tutorial_completed_at` PATCH semantics"). Do NOT add a non-null
    # constraint or a server_default here.
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
    # timestamp; the Phase 8 retake flow sends explicit `null` to clear it.
    # Three semantic states are distinguished via `model_fields_set` in the
    # service (see Task 3): (a) key absent → no-op; (b) datetime → write
    # timestamp; (c) explicit `null` → write NULL. The Pydantic annotation
    # `datetime | None = None` allows the inbound payload to be either omitted
    # or sent as `null`; the service does NOT collapse the two cases. See
    # §"Cross-plan contract — `tutorial_completed_at` PATCH semantics".
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

> **Implementer note — stale docstring at
> `src/elspeth/web/preferences/models.py:78-86`.** The existing
> docstring on `UpdateComposerPreferencesRequest` claims Pydantic v2
> cannot distinguish JSON-missing from JSON-null without a sentinel.
> That claim is stale: Pydantic v2's `model_fields_set` discriminates
> the two cases correctly under
> `ConfigDict(strict=True, extra='forbid')` (verified empirically).
> When you add `tutorial_completed_at: datetime | None = None` to
> the model in Task 2, also update the docstring to reflect that
> `banner_dismissed_at`'s "absent collapses to no-op" behaviour is a
> *deliberate spec choice* (one-way dismissal — see Phase 1B), not a
> Pydantic limitation. The new `tutorial_completed_at` field is the
> first to *use* the three-state distinction in production (absent =
> preserve, datetime = set, explicit null = clear). The Task 3
> service implementation below is the consumer of that distinction.

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


def test_explicit_null_clears_tutorial_completed_at(service):
    """Phase 8 retake contract: PATCH with explicit null nulls the column.

    Cross-plan contract — see §"Cross-plan contract — `tutorial_completed_at`
    PATCH semantics". Phase 8 Task 6's retake button PATCHes
    `{"tutorial_completed_at": null}`; the service must distinguish that
    from "field absent" (preserve) and write NULL.
    """
    stamp = datetime(2026, 5, 15, 11, 15, tzinfo=UTC)
    asyncio.run(
        service.update_composer_preferences(
            "alice-tutorial-retake",
            UpdateComposerPreferencesRequest(tutorial_completed_at=stamp),
        )
    )
    # Retake: explicit null in the PATCH body.
    result = asyncio.run(
        service.update_composer_preferences(
            "alice-tutorial-retake",
            UpdateComposerPreferencesRequest(tutorial_completed_at=None),
        )
    )
    assert result.tutorial_completed_at is None
    follow_up = asyncio.run(service.get_composer_preferences("alice-tutorial-retake"))
    assert follow_up.tutorial_completed_at is None


def test_absent_field_and_explicit_null_are_distinguished(service):
    """The service distinguishes absent-from-payload from explicit-null.

    Pydantic v2 `model_fields_set` is the discriminator. This pins the
    boundary contract that Phase 8 Task 6 depends on.
    """
    stamp = datetime(2026, 5, 15, 11, 20, tzinfo=UTC)
    asyncio.run(
        service.update_composer_preferences(
            "alice-tutorial-discriminate",
            UpdateComposerPreferencesRequest(tutorial_completed_at=stamp),
        )
    )
    # (a) Field ABSENT from payload → preserve.
    absent_payload = UpdateComposerPreferencesRequest(default_mode="freeform")
    assert "tutorial_completed_at" not in absent_payload.model_fields_set
    asyncio.run(
        service.update_composer_preferences("alice-tutorial-discriminate", absent_payload)
    )
    after_absent = asyncio.run(
        service.get_composer_preferences("alice-tutorial-discriminate")
    )
    assert after_absent.tutorial_completed_at == stamp  # preserved

    # (c) Field PRESENT with explicit null → write NULL.
    null_payload = UpdateComposerPreferencesRequest(tutorial_completed_at=None)
    assert "tutorial_completed_at" in null_payload.model_fields_set
    asyncio.run(
        service.update_composer_preferences("alice-tutorial-discriminate", null_payload)
    )
    after_null = asyncio.run(
        service.get_composer_preferences("alice-tutorial-discriminate")
    )
    assert after_null.tutorial_completed_at is None


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
            # Three semantic states (see Cross-plan contract):
            #   (a) field absent from payload         → preserve existing value
            #   (b) field present, datetime value     → write the timestamp
            #   (c) field present, value is None      → write NULL (Phase 8 retake)
            # Pydantic v2 exposes `model_fields_set` containing only the keys
            # that were explicitly provided by the caller; this is how we
            # distinguish (a) "key absent" from (c) "key present but null"
            # without inventing a sentinel.
            tutorial_was_provided = "tutorial_completed_at" in payload.model_fields_set
            if tutorial_was_provided:
                resolved_tutorial = payload.tutorial_completed_at  # may be a datetime or None
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
                # Insert-side: use the resolved value so a fresh-row upsert
                # without `tutorial_completed_at` in the payload writes NULL
                # (the default for new users) rather than a stale-read value
                # from an earlier transaction.
                "tutorial_completed_at": resolved_tutorial,
                "updated_at": now,
            }
            stmt = sqlite_insert(user_preferences_table).values(**values)
            update_clause: dict[str, object] = {"updated_at": now}
            # Note: default_mode and banner_dismissed_at retain the Phase 1A
            # "None = preserve" convention (there is no client need to NULL
            # either field via PATCH; the banner-dismissed timestamp is
            # write-once-and-leave-alone). Only `tutorial_completed_at` uses
            # the three-state `model_fields_set` discrimination because
            # Phase 8 Task 6's retake requires the explicit-null write path.
            if payload.default_mode is not None:
                update_clause["default_composer_mode"] = payload.default_mode
            if payload.banner_dismissed_at is not None:
                update_clause["banner_dismissed_at"] = payload.banner_dismissed_at
            if tutorial_was_provided:
                # Writes either a datetime (set) or NULL (Phase 8 retake).
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

Expected: PASS — all service tests green (existing + 7 new — 5 originally specified plus 2 added for the cross-plan retake contract).

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


def test_patch_with_explicit_null_clears_tutorial(client_as_alice: TestClient) -> None:
    """Phase 8 retake contract at the HTTP boundary.

    See §"Cross-plan contract — `tutorial_completed_at` PATCH semantics".
    Phase 8 Task 6 PATCHes `{"tutorial_completed_at": null}` to retrigger
    the tutorial; the route must accept the body and return a payload with
    `tutorial_completed_at = null`.
    """
    # First set the field via a normal finalisation PATCH.
    set_response = client_as_alice.patch(
        "/api/composer-preferences",
        json={"tutorial_completed_at": "2026-05-15T12:40:00Z"},
    )
    assert set_response.status_code == 200
    assert set_response.json()["tutorial_completed_at"] == "2026-05-15T12:40:00Z"
    # Phase 8 retake: explicit null clears the column.
    clear_response = client_as_alice.patch(
        "/api/composer-preferences",
        json={"tutorial_completed_at": None},
    )
    assert clear_response.status_code == 200
    assert clear_response.json()["tutorial_completed_at"] is None
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

The cache stores the **deterministic output content** of a canonical-seed
run, keyed on `(canonical_prompt, model)`. **The cache never stores identity
references** (no `run_id`, no `session_id`, no `interpretation_event_id`)
from the cache-seeding run — those are owned by the user/session that
seeded the cache and have no meaning to a different user replaying the
same content. On a cache hit the run-path synthesises a fresh Landscape
entry under the **current** session, populated from the cached content
plus a `seeded_from_cache: true` provenance marker (Task 7). The cache key
is recorded on the marker so an auditor can join across runs that share
the same cache-seeded content.

The cached fields are:

- `rows` — the row-level LLM rating output (list of dicts shaped per the
  canonical pipeline's terminal node).
- `source_data_hash` — SHA-256 of the source URL set, computed by the
  source plugin at run time. Reproducible across runs of the same input,
  so the cache-replayed run's Landscape entry carries the same hash as
  the original seeding run did — genuine determinism evidence, not a
  copied identity field.
- `llm_call_count` — integer count of LLM calls the cache-seeding run
  made. Persisted so the replay's Landscape entry can record an
  authentic count (the replay itself records `llm_call_count = 0`; the
  seeding-run count is exposed separately for cost-attribution copy in
  turn 5).
- `pipeline_yaml` — the canonical pipeline YAML the cache-seeding run
  executed. Persisted so replay records the same pipeline definition
  the seeded content was generated against; otherwise a hit could
  replay against a drifted pipeline and produce inconsistent audit
  evidence.

**Cache directory:** `~/.elspeth_web/tutorial_cache/`. Operators can override
via a constructor argument (used by tests).

**File naming:** `<sha256_hex>.json` where the hex is the SHA-256 of
`f"{canonical_prompt}:{model_id}"`. The plain canonical prompt and model
are also stored inside the JSON for diagnostic visibility (an operator
inspecting a file should be able to confirm what it caches without
recomputing the hash).

**Operational guarantees** (corruption discipline; tier classification is
deferred to P23 which decides whether "server-generated cache content"
warrants Tier-1 framing):

- A file present → must parse via Pydantic. Parse failure → crash. No
  fallback to a live run: that would mask corruption.
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
from pydantic import ValidationError

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


_CANONICAL_PIPELINE_YAML = """\
source:
  type: inline_blob
  rows:
    - url: ato.gov.au
transforms:
  - type: web_scrape
  - type: llm_rate
sink:
  type: tutorial_summary
"""


def test_lookup_returns_entry_on_hit(cache: TutorialCache) -> None:
    entry = TutorialCacheEntry(
        canonical_prompt=CANONICAL_SEED_PROMPT,
        model_id="claude-opus-4-7",
        cached_at=datetime(2026, 5, 15, tzinfo=UTC),
        rows=[{"url": "ato.gov.au", "score": 5, "rationale": "clear nav"}],
        source_data_hash="a7f3e2deadbeef",
        llm_call_count=5,
        pipeline_yaml=_CANONICAL_PIPELINE_YAML,
    )
    cache.store(entry)
    got = cache.lookup(CANONICAL_SEED_PROMPT, "claude-opus-4-7")
    assert got is not None
    assert got.canonical_prompt == CANONICAL_SEED_PROMPT
    assert got.model_id == "claude-opus-4-7"
    assert got.source_data_hash == "a7f3e2deadbeef"
    assert got.llm_call_count == 5
    assert got.rows[0]["url"] == "ato.gov.au"
    assert got.pipeline_yaml == _CANONICAL_PIPELINE_YAML


def test_entry_rejects_identity_fields() -> None:
    """Architectural invariant: the cache schema MUST NOT accept run_id or
    interpretation_event_id. Foreign identity must not enter the cache."""
    with pytest.raises(ValidationError):
        TutorialCacheEntry.model_validate({
            "canonical_prompt": CANONICAL_SEED_PROMPT,
            "model_id": "claude-opus-4-7",
            "cached_at": "2026-05-15T00:00:00+00:00",
            "rows": [],
            "source_data_hash": "hash",
            "llm_call_count": 0,
            "pipeline_yaml": _CANONICAL_PIPELINE_YAML,
            "run_id": "abc-123",  # extra field — must be rejected
        })


def test_lookup_misses_on_different_model(cache: TutorialCache) -> None:
    entry = TutorialCacheEntry(
        canonical_prompt=CANONICAL_SEED_PROMPT,
        model_id="claude-opus-4-7",
        cached_at=datetime(2026, 5, 15, tzinfo=UTC),
        rows=[],
        source_data_hash="hash",
        llm_call_count=0,
        pipeline_yaml=_CANONICAL_PIPELINE_YAML,
    )
    cache.store(entry)
    assert cache.lookup(CANONICAL_SEED_PROMPT, "claude-opus-4-8") is None


def test_lookup_misses_on_different_prompt(cache: TutorialCache) -> None:
    entry = TutorialCacheEntry(
        canonical_prompt=CANONICAL_SEED_PROMPT,
        model_id="claude-opus-4-7",
        cached_at=datetime(2026, 5, 15, tzinfo=UTC),
        rows=[],
        source_data_hash="hash",
        llm_call_count=0,
        pipeline_yaml=_CANONICAL_PIPELINE_YAML,
    )
    cache.store(entry)
    edited = CANONICAL_SEED_PROMPT + " and also rate accessibility"
    assert cache.lookup(edited, "claude-opus-4-7") is None


def test_store_and_lookup_round_trip(cache: TutorialCache, cache_dir: Path) -> None:
    entry = TutorialCacheEntry(
        canonical_prompt=CANONICAL_SEED_PROMPT,
        model_id="claude-opus-4-7",
        cached_at=datetime(2026, 5, 15, tzinfo=UTC),
        rows=[{"url": "example.gov.au", "score": 3}],
        source_data_hash="hash",
        llm_call_count=5,
        pipeline_yaml=_CANONICAL_PIPELINE_YAML,
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
    # No identity fields written to disk.
    assert "run_id" not in raw
    assert "interpretation_event_id" not in raw


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
        "rows": [],
        "source_data_hash": "hash",
        "llm_call_count": 0,
        "pipeline_yaml": _CANONICAL_PIPELINE_YAML,
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
        rows=[{"url": "example.gov.au"}],
        source_data_hash="hash",
        llm_call_count=0,
        pipeline_yaml=_CANONICAL_PIPELINE_YAML,
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
    """Cached deterministic output of a canonical-seed run.

    Content-not-identity invariant: this model stores the output content of a
    canonical-seed-prompt run, never identity references (no run_id,
    session_id, user_id, or interpretation_event_id) from the cache-seeding
    run. On a cache hit the run-path synthesises a fresh Landscape entry
    under the **current** session populated from these fields plus a
    `seeded_from_cache: true` marker; the original seeding-run identity is
    referenced only indirectly, via the cache key, so an auditor can join
    across runs that share the same cache-seeded content.

    Field semantics:

    - ``rows``: row-level LLM rating output, shape = canonical pipeline's
      terminal-node output.
    - ``source_data_hash``: SHA-256 (hex) of the source URL set, computed
      by the source plugin. Reproducible across runs over the same input,
      so the replayed Landscape entry can carry this value as genuine
      determinism evidence rather than a copied identity field.
    - ``llm_call_count``: number of LLM calls the cache-seeding run made.
      Persisted so turn 5's cost-attribution copy can cite a real number;
      the replay's own Landscape entry records ``llm_call_count = 0``
      (the cache served the responses).
    - ``pipeline_yaml``: canonical pipeline YAML the cache-seeding run
      executed. Persisted so a hit replays against the same pipeline
      definition the content was generated against.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    canonical_prompt: str
    model_id: str
    cached_at: datetime
    rows: list[dict[str, Any]]
    source_data_hash: str
    llm_call_count: int
    pipeline_yaml: str


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
4. **On hit: replays the cached content against the current session** by
   synthesising a real Landscape entry under the current user's `session_id`
   with a freshly-minted `run_id` owned by the current session. The
   synthesised entry records:
   - `pipeline_yaml` from the cache entry (so the replay's audit trail
     references the same pipeline definition the cached content was
     generated against);
   - `rows` exactly as cached (no re-execution; this is the optimisation);
   - `source_data_hash` from the cache entry (reproducible across runs
     over the same input — this is determinism evidence, not a copied
     identity);
   - `llm_call_count = 0` on the new run (the cache served the LLM
     responses);
   - `seeded_from_cache: true` provenance marker carrying the cache key
     (`SHA-256(canonical_prompt + ":" + model_id)`) so an auditor can
     join across runs that share the same cache-seeded content; this
     surfaces the cache-replay in the audit trail rather than hiding it.

   Returns `run_id` (the new, current-session-owned identifier) to the
   frontend. The frontend's turn 5 audit-story call then targets the
   **current** session's run — a same-ownership query, with the
   `seeded_from_cache` marker visible in the response so turn 5 can
   acknowledge the cache-replay in user-facing copy.
5. On miss: proceeds with the normal run path. **After** the run completes
   successfully, calls `app.state.tutorial_cache.store(...)` to populate
   the cache for the next user. (P18 will gate this write on the
   all-rows-succeeded condition; see the `# TODO: P18` hook in Step 4.)

**Edge: a user edits the seed.** The canonical-seed-match check fails; the
cache is bypassed entirely; the user pays for a live LLM run. This is the
intended behaviour — caching edited prompts would require a much larger key
space and create per-edit cache churn.

**Cross-ownership query is impossible by construction.** The cache stores
no foreign identity, so the run-path cannot return a foreign `run_id`. The
frontend therefore cannot accidentally query another session's audit
endpoint. This is the architectural fix that the P2 review surfaced.

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


_CANONICAL_PIPELINE_YAML = """\
source:
  type: inline_blob
  rows:
    - url: ato.gov.au
transforms:
  - type: web_scrape
  - type: llm_rate
sink:
  type: tutorial_summary
"""


def _seed_canonical_cache_entry(app: FastAPI) -> None:
    """Helper: pre-populate the cache with a recognisable canonical entry."""
    app.state.tutorial_cache.store(
        TutorialCacheEntry(
            canonical_prompt=CANONICAL_SEED_PROMPT,
            model_id="claude-opus-4-7",
            cached_at=datetime(2026, 5, 15, tzinfo=UTC),
            rows=[{"url": "ato.gov.au", "score": 5, "rationale": "clear nav"}],
            source_data_hash="a7f3e2cached",
            llm_call_count=5,
            pipeline_yaml=_CANONICAL_PIPELINE_YAML,
        )
    )


def test_cache_hit_returns_cached_content_without_calling_llm(
    app_with_cache: FastAPI, cache_dir: Path
) -> None:
    """User in tutorial mode + canonical seed + cache hit → no LLM call."""
    _seed_canonical_cache_entry(app_with_cache)

    # The user is in tutorial mode (no PATCH yet → tutorial_completed_at is None).
    client = TestClient(app_with_cache)

    # Patch the LLM-call path so we can assert it was NOT called.
    with patch("<llm-call-path>") as mock_llm:
        response = client.post(
            "/api/sessions/<session_id>/runs",
            json={"<canonical-seed-pipeline-shape>": "..."},
        )
        assert response.status_code == 200
        body = response.json()
        # The cached rows are returned verbatim.
        assert body["rows"][0]["url"] == "ato.gov.au"
        # The cached source_data_hash propagates to the run response.
        assert body["source_data_hash"] == "a7f3e2cached"
        # The LLM was never called.
        mock_llm.assert_not_called()


def test_cache_hit_creates_landscape_entry_under_current_session(
    app_with_cache: FastAPI, cache_dir: Path
) -> None:
    """Architectural invariant: a cache hit produces a NEW run_id owned by
    the current user's session — not a foreign run_id from the cache-seeding
    session.

    This is the P2-review fix: cross-ownership audit-story queries are
    impossible because the cache never stores foreign identity.
    """
    _seed_canonical_cache_entry(app_with_cache)

    client = TestClient(app_with_cache)
    response = client.post(
        "/api/sessions/<session_id>/runs",
        json={"<canonical-seed-pipeline-shape>": "..."},
    )
    assert response.status_code == 200
    body = response.json()
    returned_run_id = body["run_id"]

    # The audit-story endpoint resolves under the CURRENT session — no
    # cross-ownership query is possible.
    story = client.get(
        f"/api/sessions/<session_id>/runs/{returned_run_id}/audit-story"
    )
    assert story.status_code == 200
    story_body = story.json()
    # The seeded_from_cache marker surfaces the cache-replay provenance.
    assert story_body["seeded_from_cache"] is True
    # The cache key is recorded so an auditor can join to the seeding run.
    assert isinstance(story_body["cache_key"], str)
    assert len(story_body["cache_key"]) == 64  # SHA-256 hex
    # The replayed Landscape entry records llm_call_count = 0 (no live LLM).
    assert story_body["llm_call_count"] == 0


def test_cache_hits_for_different_users_produce_distinct_run_ids(
    app_with_cache: FastAPI, cache_dir: Path
) -> None:
    """Two users hitting the same cached content get two distinct run_ids,
    each owned by the issuing user's session.

    This is the second leg of the content-not-identity invariant: cached
    content is shared, but the audit identity of each replay is unique.
    """
    _seed_canonical_cache_entry(app_with_cache)

    client_alice = TestClient(app_with_cache)
    response_alice = client_alice.post(
        "/api/sessions/<alice_session_id>/runs",
        json={"<canonical-seed-pipeline-shape>": "..."},
    )
    assert response_alice.status_code == 200
    run_id_alice = response_alice.json()["run_id"]

    # Switch to Bob (test-fixture detail filled in during recon; the
    # dependency-override for get_current_user is reassigned).
    client_bob = TestClient(app_with_cache)
    response_bob = client_bob.post(
        "/api/sessions/<bob_session_id>/runs",
        json={"<canonical-seed-pipeline-shape>": "..."},
    )
    assert response_bob.status_code == 200
    run_id_bob = response_bob.json()["run_id"]

    assert run_id_alice != run_id_bob


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
    # After the run, the cache should contain one entry — and that entry
    # carries content, not identity.
    files = list(cache_dir.iterdir())
    assert len(files) == 1
    import json as _json
    raw = _json.loads(files[0].read_text())
    assert "rows" in raw
    assert "source_data_hash" in raw
    assert "llm_call_count" in raw
    assert "pipeline_yaml" in raw
    assert "run_id" not in raw  # invariant
    assert "interpretation_event_id" not in raw  # invariant


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
    app_with_cache.state.tutorial_cache.store(
        TutorialCacheEntry(
            canonical_prompt=CANONICAL_SEED_PROMPT,
            model_id="claude-opus-4-7",
            cached_at=datetime(2026, 5, 15, tzinfo=UTC),
            rows=[{"url": "cached-url", "score": 99}],
            source_data_hash="should-not-appear",
            llm_call_count=5,
            pipeline_yaml=_CANONICAL_PIPELINE_YAML,
        )
    )

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
    app_with_cache.state.tutorial_cache.store(
        TutorialCacheEntry(
            canonical_prompt=CANONICAL_SEED_PROMPT,
            model_id="claude-opus-4-7",
            cached_at=datetime(2026, 5, 15, tzinfo=UTC),
            rows=[],
            source_data_hash="cached-hash",
            llm_call_count=0,
            pipeline_yaml=_CANONICAL_PIPELINE_YAML,
        )
    )

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
from elspeth.web.preferences.tutorial_cache import CANONICAL_SEED_PROMPT, _compute_key

async def execute_pipeline_run(
    request: Request,
    user: UserIdentity,
    session_id: str,
    pipeline_state: dict,
) -> dict:
    """Execute a pipeline run for the user.

    Phase 4: tutorial-mode users running the canonical seed pipeline consult
    the cache first. On hit, replay the cached **content** against a new
    Landscape entry owned by the current session (no foreign identity is
    returned). On miss, run live; on success, populate the cache with the
    deterministic content of the just-completed run.
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
                # Cache hit — REPLAY the cached content under the current
                # user's session. A fresh Landscape entry is created with a
                # new run_id owned by `session_id`; the cache key is
                # recorded on the seeded_from_cache marker so the audit
                # trail surfaces the cache-replay rather than hiding it.
                return await _replay_cached_content_to_landscape(
                    request=request,
                    user=user,
                    session_id=session_id,
                    cache_entry=cache_entry,
                    cache_key=_compute_key(CANONICAL_SEED_PROMPT, model_id),
                )

    # Normal path: run live.
    result = await _execute_pipeline_live(request, user, session_id, pipeline_state)

    # Tutorial-mode + canonical-seed cache populate (on success only).
    # TODO: P18 — replace `_is_successful_run` with an all-rows-succeeded
    # gate so a partial-success run (some rows quarantined) does not poison
    # the cache for the next user. The hook is intentionally narrow: the
    # write site is here, the gate condition is the only change.
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
                rows=result["rows"],
                source_data_hash=result["source_data_hash"],
                llm_call_count=result["llm_call_count"],
                pipeline_yaml=result["pipeline_yaml"],
            )
        )

    return result


async def _replay_cached_content_to_landscape(
    *,
    request: Request,
    user: UserIdentity,
    session_id: str,
    cache_entry: TutorialCacheEntry,
    cache_key: str,
) -> dict:
    """Synthesise a Landscape entry under ``session_id`` from cached content.

    Creates a new ``run_id`` owned by the current session and records:

    - ``pipeline_yaml`` from the cache entry (replay uses the same pipeline
      definition the seeded content was generated against);
    - ``rows`` and ``source_data_hash`` exactly as cached (the optimisation);
    - ``llm_call_count = 0`` (no live LLM calls were made by this replay);
    - ``seeded_from_cache = True`` plus ``cache_key`` as metadata, so an
      auditor querying ``explain(recorder, run_id, token_id)`` sees the
      cache-replay provenance and can join to the original seeding run via
      ``cache_key``.

    No foreign identity (no ``run_id``, ``session_id``, ``user_id`` from
    the cache-seeding session) is returned or recorded.
    """
    # The exact Landscape write API depends on recon; the implementation
    # uses the same record-a-completed-run path that `_execute_pipeline_live`
    # would have used, but feeds it cached content instead of live results.
    new_run_id = await request.app.state.landscape.record_synthesised_run(
        session_id=session_id,
        user_id=user.user_id,
        pipeline_yaml=cache_entry.pipeline_yaml,
        rows=cache_entry.rows,
        source_data_hash=cache_entry.source_data_hash,
        llm_call_count=0,
        metadata={
            "seeded_from_cache": True,
            "cache_key": cache_key,
            "cache_seeding_llm_call_count": cache_entry.llm_call_count,
        },
    )
    return {
        "run_id": new_run_id,
        "source_data_hash": cache_entry.source_data_hash,
        "rows": cache_entry.rows,
    }
```

The `_is_canonical_seed_pipeline`, `_model_id_for_pipeline`,
`_is_successful_run`, and `_execute_pipeline_live` helpers are defined in
the same file. Their shape depends on recon; the executor fills them in
based on the actual run-path internals. `_replay_cached_content_to_landscape`
replaces the old `_cached_entry_to_run_response` — the substantive change
is that it writes a real Landscape entry under the current session rather
than synthesising a response dict containing a foreign `run_id`.

**Operational discipline:** cache `lookup`/`store` crash on corruption
(see Task 5 "Operational guarantees"); `_is_canonical_seed_pipeline` reads
Tier-2 `pipeline_state` (no-match is normal, corrupt structure crashes);
`model_id` extraction crashes on absence (it's required by every live run
too). The synthesised Landscape entry is written before the response is
returned; if the write fails, the request fails — there is no
silent-cache-hit fallback. Per CLAUDE.md "no defensive programming": no
try/except wrapping of the replay path.

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

**Phase 8 retake mechanism (co-owned contract).** Phase 8 Task 6 ships a
"Replay hello-world tutorial" button that nulls `tutorial_completed_at` via
`PATCH /api/composer-preferences` with body `{"tutorial_completed_at": null}`.
The schema (`nullable=True`), the Pydantic model (`datetime | None`), the
service (absent-vs-null discrimination via `model_fields_set`), and the route
(no structural change) shipped by Phase 4 are all the preconditions for that
mechanism. See §"Cross-plan contract — `tutorial_completed_at` PATCH
semantics" near the top of this document. The audit emit for the retake
event itself lives in Phase 8; this plan does not change
`composer.preferences.patch_total` or its emit site.

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

### 2026-05-19 — cross-plan contract amendment (Phase 4 ↔ Phase 8)

Pass-1 review of Phase 4 surfaced a Systems S1 contract rupture against
Phase 8 Task 6's retake mechanism: Phase 4 originally disallowed
nullification of `tutorial_completed_at` via PATCH (the Pydantic field
default was treated as "leave alone" and operators were told to SQL-UPDATE
directly), while Phase 8 expected to clear the column via the same PATCH
endpoint. Synthesizer-adopted resolution: Option (a) — allow nullification
via explicit `null` in the PATCH body; distinguish absent-from-payload from
explicit-null via Pydantic v2's `model_fields_set`. Same field, same column,
single shared contract co-owned by Phases 4 and 8.

Edits applied:

- §Scope boundaries `update_composer_preferences` bullet rewritten to name
  the three semantic states (absent / datetime / null).
- New §"Cross-plan contract — `tutorial_completed_at` PATCH semantics"
  section inserted between §"DB-delete cadence" and §"New endpoints".
- Task 1 column-add prose: explicit "do not add NOT NULL or server_default"
  note attached to the `nullable=True` line.
- Task 2 model comment rewritten to document the three-state contract and
  the `model_fields_set` discrimination pattern.
- Task 3 service code: absent-vs-null distinguished via `model_fields_set`
  for `tutorial_completed_at` only. `default_mode` and `banner_dismissed_at`
  retain the Phase 1A "None = preserve" convention (no client need to NULL
  either field).
- Task 3 tests: two new tests (`test_explicit_null_clears_tutorial_completed_at`,
  `test_absent_field_and_explicit_null_are_distinguished`).
- Task 4 tests: one new test (`test_patch_with_explicit_null_clears_tutorial`).
- §Forward compatibility: new paragraph naming the Phase 8 retake mechanism
  and citing the shared contract block.

Co-edits applied in `21-phase-4-hello-world-tutorial.md` (Open Question C3
resolution updated, vocabulary block extended, file inventory pointer) and
`20-phase-8-polish-and-telemetry.md` (Task 6 PATCH body changed from
`{"tutorial_completed": false}` to `{"tutorial_completed_at": null}`,
Trust-tier check updated, telemetry-primacy footnote updated, retake
audit-emit boundary surfaced for Phase 8 reviewers).
