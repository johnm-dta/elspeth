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

### Phase 4 lifecycle: two DB-delete events on two distinct databases

Phase 4A as currently scoped lands **two** schema additions on **two
different SQLite databases**, each carrying its own SCHEMA_EPOCH constant
and its own `_assert_schema_version()` startup guard. Both require a
DB-delete on staging; the operator must perform each delete on the
correct file. Treating them as one event will leave the other DB in a
silently-broken state.

| # | Commit | Database file | Schema-epoch constant | Schema change | Operator action |
|---|---|---|---|---|---|
| 1 | Task 1 (this plan, `feat(web): add tutorial_completed_at…`) | **Sessions DB** (`src/elspeth/web/sessions/models.py` schema; configured `composer.sessions_db_path`, default `./sessions.db`) | `SESSION_SCHEMA_EPOCH` in `src/elspeth/web/sessions/models.py` (bumped by this commit) | `user_preferences_table` gains `tutorial_completed_at: datetime | None`. | Delete the sessions DB file before the new code serves traffic. All users' `tutorial_completed_at` reset to NULL — every user retakes the tutorial on next login. |
| 2 | Task 7.0 (this plan, `feat(landscape): extend runs_table with audit-story columns…`) | **Landscape DB** (`src/elspeth/core/landscape/schema.py`; configured `audit_database_path`) | `SQLITE_SCHEMA_EPOCH` in `src/elspeth/core/landscape/schema.py` (bumped by Task 7.0 — separate scope) | `runs_table` gains six audit-story columns. | Delete the Landscape audit DB file. All historical run audit data is lost on staging; production deferred until Phase 9 migration runner. |

The two events are independent because the two DBs are independent — the
Landscape audit DB and the sessions DB have separate files, separate
lifecycles, and separate epoch constants (see the comment block above
`SESSION_SCHEMA_EPOCH` in `src/elspeth/web/sessions/models.py`, which
calls this out explicitly). Deleting one does not affect the other; both
must be performed when both commits ship.

**Why both deletes are mandatory even if the column-add looks additive:**
without the epoch bump, an existing DB silently fails on first access to
the new column. PRAGMA `user_version` still matches `SCHEMA_EPOCH` (no
mismatch is raised at startup), but the column itself is absent — the
operator sees a confusing runtime `OperationalError: no such column` on
the first request that touches it, with no startup signal pointing at
schema drift. The epoch bump is what converts that silent runtime
failure into an explicit "delete your DB and restart" startup abort,
which is the canonical operator-visible signal in this project.

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

Consumed by Phase 4B Part 2. Defined here so 21b2 can reference them. Implementation tasks: **Task 7.1** (POST /api/tutorial/run), **Task 7.2** (GET …/audit-story), **Task 7.3** (frontend `client.ts` functions consuming both). Tasks 7.1–7.3 follow Task 7 in this document.

**`POST /api/tutorial/run`** — body: `{"session_id": "<uuid>", "prompt": "<canonical-or-edited-seed>"}`. Response: `{"run_id": "<uuid>", "output": {"rows": [...], "source_data_hash": "<hex>"}, "seeded_from_cache": <bool>, "cache_key": "<hex>" | null}`. Cache-hit: the backend **synthesises a real Landscape entry under the current user's session** via `_replay_cached_content_to_landscape` (defined in Task 7), populated from the cached content (rows, source_data_hash, llm_call_count=0, pipeline_yaml) plus a `seeded_from_cache: true` marker carrying the cache key. The returned `run_id` is **owned by the current session** — there is no foreign-run reference in the response. Cache-miss: live run (~30s), populates cache on success. Cache is bypassed (live run, no consult, no write) when the user's `tutorial_completed_at IS NOT NULL` (post-completion) **or** when their `default_mode == 'freeform'` (freeform users skip tutorial caching entirely). Unknown session → 404. Tier-1 corruption on the caller's preferences row → 500 (`CorruptPreferencesError` propagates to the global handler). Defined by Task 7.1.

**`GET /api/sessions/{session_id}/runs/{run_id}/audit-story`** — response: `{"run_id": "<uuid>", "session_id": "<uuid>", "llm_call_count": N, "output_file_hash": "<hex>", "run_started_at": "<iso8601>", "plugin_versions": {...}, "seeded_from_cache": <bool>, "cache_key": "<hex>" | null}`. Reads **entirely from real Landscape audit rows** — **no field is ever synthesised or defaulted**. When the run was a cache hit, `seeded_from_cache` is `true`, `llm_call_count` is `0`, and `cache_key` is the SHA-256 that points at the original cache-seeding run for cross-run lineage joins. Run not in session → 404. Session not owned by caller → 404 (IDOR contract: never 403, to avoid leaking session existence — see `src/elspeth/web/sessions/ownership.py:33`). Tier-1 corruption (audit row missing a required field such as `llm_call_count`) → 500 (named exception). Landscape failure propagates — no fallback (design doc 04: "Otherwise the demonstration is theatre."). Defined by Task 7.2.

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
- `src/elspeth/web/composer/tutorial_run_routes.py` — POST /api/tutorial/run (Task 7.1).
- `src/elspeth/web/composer/tutorial_service.py` — tutorial-run service (Task 7.1).
- `tests/integration/web/test_tutorial_routes.py` — tutorial-route integration tests (Task 7.1).
- `src/elspeth/web/sessions/audit_story_service.py` — audit-story service (Task 7.2).
- `tests/integration/web/test_audit_story_routes.py` — audit-story integration tests (Task 7.2).
- `src/elspeth/web/frontend/src/api/client.tutorial.test.ts` — frontend client tests (Task 7.3).

**Modified:**

- `src/elspeth/web/sessions/models.py` — add `tutorial_completed_at` column.
- `src/elspeth/web/preferences/models.py` — extend Pydantic models.
- `src/elspeth/web/preferences/service.py` — extend read/write code paths.
- `tests/unit/web/preferences/test_schema.py` — extend expected-columns set.
- `tests/unit/web/preferences/test_models.py` — extend Pydantic tests.
- `tests/unit/web/preferences/test_service.py` — extend service tests.
- `tests/integration/web/test_preferences_routes.py` — extend route tests.
- The composer run-path file (identified during Task 7) — wire cache consult; Task 7.1 extends `_is_canonical_seed_pipeline` with a force-live escape.
- `src/elspeth/web/sessions/routes.py` — add `GET …/audit-story` handler (Task 7.2).
- `src/elspeth/web/sessions/schemas.py` — add `RunAuditStoryResponse` (Task 7.2).
- `src/elspeth/web/frontend/src/api/client.ts` — add `runTutorialPipeline`, `getRunAuditSummary`; rename `updateSessionTitle` → `renameSession` (Task 7.3).
- Frontend call sites of the renamed function (discovered during Task 7.3 Step 1).
- The FastAPI app-composition site — `include_router(create_tutorial_run_router())` (Task 7.1).

**Not modified:**

- `src/elspeth/web/preferences/routes.py` — Pydantic-model extension propagates
  automatically through `response_model`.

## Database migration note (operator action)

Task 1 and Task 7.0 each require a DB-delete before new code serves
traffic — Task 1 on the **sessions DB** (because `SESSION_SCHEMA_EPOCH`
bumps and `user_preferences_table` gains a column), Task 7.0 on the
**Landscape audit DB** (because `runs_table` gains six columns).
Phase 4B's smoke task performs both. If Phase 4A ships independently,
operator must perform both deletes first; they are independent
operations on independent files. All users' `tutorial_completed_at`
resets to NULL — every user retakes the tutorial on next login. See
§"DB-delete cadence" for the full sequence context, including the
table enumerating both events.

## Verification approach

Each task is TDD-shaped (failing test, run-to-fail, implement, run-to-pass,
commit). After Tasks 1–7.3 land, the Phase 4B integration tests and Playwright
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
- Modify: `src/elspeth/web/sessions/models.py` (epoch bump + epoch-history comment + column add).
- Modify: `tests/unit/web/preferences/test_schema.py`.

> **OPERATOR ACTION required at commit time.** This commit is the first
> of the two Phase 4 DB-delete events (see §"DB-delete cadence: Phase 4
> lifecycle: two DB-delete events on two distinct databases"). After
> merging this commit and before any downstream Phase 4 task runs on
> staging, the operator must delete the **sessions DB file** on staging
> (`composer.sessions_db_path`, default `./sessions.db`). The
> `SESSION_SCHEMA_EPOCH` bump in Step 0a below will trigger an
> explicit startup abort with a "Session database schema does not
> match SESSION_SCHEMA_EPOCH=…" message on existing DBs — that abort
> is the canonical operator-visible signal, by design. Do not attempt
> to skip the abort by reverting the bump; the bump is what makes the
> drift detectable. Surface this in the commit message and announce
> it before staging deploy. Cross-link: project memory
> `feedback_operator_gate_destructive_actions`.

The first three sub-steps (Step 0a, 0b, 0c) handle the
`SESSION_SCHEMA_EPOCH` bump and epoch-history comment extension; they
land in the **same commit** as the column-add so production can never
observe a state where the column was added without the epoch having
been bumped (which would defeat the startup-abort signal). Sub-steps
ordered before Step 1 deliberately — the implementer must update the
epoch *first*, so the schema-drift signal lands together with the
column it announces.

- [ ] **Step 0a: Bump `SESSION_SCHEMA_EPOCH` from 5 to 6.**

Open `src/elspeth/web/sessions/models.py`. The constant currently reads
(verify the exact value before editing — bump from whatever is live):

```python
SESSION_SCHEMA_EPOCH = 5
```

Change to:

```python
SESSION_SCHEMA_EPOCH = 6
```

This is atomic — no "support both 5 and 6" branches (No Legacy Code
policy, CLAUDE.md). The bump must land in the **same commit** as the
column-add in Step 3 below; a commit that adds the column without
bumping the epoch leaves a window where existing sessions DBs are
silently broken (the column is absent on disk but PRAGMA
`user_version` still equals the constant, so
`_assert_schema_version()` raises no mismatch and the operator gets a
confusing `OperationalError: no such column: tutorial_completed_at` at
first column access instead of the canonical startup abort).

- [ ] **Step 0b: Extend the epoch-history comment.**

Immediately above `SESSION_SCHEMA_EPOCH = …` there is a comment block
documenting each epoch bump (see the live file: the block starts
`# Epoch history (pre-1.0 policy — bumps require DB recreation):` and
already enumerates epochs 1 through 5). Append a new entry for
epoch 6, preserving the existing indentation, prose style, and "→"
arrow convention used by earlier entries:

```python
#   6 → user_preferences_table gains tutorial_completed_at: datetime | None
#        (Phase 4A, 2026-05-19). Column is nullable with no server
#        default; non-NULL means the user completed the hello-world
#        tutorial. Phase 8 Task 6's retake button PATCHes the field
#        back to NULL — the column is load-bearing for that contract
#        (see plan §"Cross-plan contract — `tutorial_completed_at`
#        PATCH semantics" in `21a-phase-4-backend.md`). Operators
#        upgrading across this boundary MUST delete their session DB.
```

Do not modify the existing entries for epochs 1–5; this is an
append-only history. The format above mirrors epoch-2's "operators
upgrading across this boundary MUST delete" phrasing intentionally —
that prose is the canonical operator-action signal in this file.

- [ ] **Step 0c: Verify the existing schema-validator branch is unchanged.**

Open `src/elspeth/web/sessions/schema.py`. Confirm
`_assert_schema_version()` exists and reads PRAGMA `user_version`,
comparing against the imported `SESSION_SCHEMA_EPOCH`. No code change
is required here — the validator picks up the new constant value
automatically. This sub-step exists so the implementer affirms the
mechanism is in place before relying on it. If the validator is
missing or its semantics have changed since this plan was written,
stop and surface to the operator before proceeding to Step 1.

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
git commit -m "$(cat <<'EOF'
feat(web): add tutorial_completed_at to user_preferences (Phase 4A.1)

Bumps SESSION_SCHEMA_EPOCH 5 → 6 and adds tutorial_completed_at to
user_preferences_table. The epoch bump and the column add land in the
same commit so the operator-visible startup-abort signal
(_assert_schema_version) fires correctly on existing DBs — adding the
column without bumping the epoch would leave existing sessions DBs
silently broken until first column access.

OPERATOR ACTION: delete the sessions DB on staging before deploying
this commit. See `21a-phase-4-backend.md` §"DB-delete cadence:
Phase 4 lifecycle: two DB-delete events on two distinct databases".
EOF
)"
```

The commit message must surface the OPERATOR ACTION explicitly so the
deploy-time reader (human or automation reviewing the merge) sees the
DB-delete requirement without having to read the plan doc. This is
the first of two Phase 4 DB-delete events; Task 7.0's commit must
carry an analogous notice for the Landscape audit DB.

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

> **Step 1 prerequisite — counter-emit test plumbing. Before implementing
> the new tests, you must:** introduce an OTel `InMemoryMetricReader`
> fixture for the preferences-service test module AND define the
> `_last_patch_counter_attributes()` helper that the two new
> counter-emit tests below depend on. The two counter-emit tests
> (`test_patch_tutorial_emits_counter_with_tutorial_changed_label`,
> `test_patch_without_tutorial_emits_counter_with_tutorial_changed_false`)
> reference this fixture and helper by name; without them the tests
> cannot be written, and a placeholder helper that returns a hand-built
> dict would pass trivially while defeating the purpose of testing
> guard #3 (`_PREFERENCES_PATCH_COUNTER` emit).
>
> **Reference pattern in the live codebase:** the canonical
> `InMemoryMetricReader` rebind pattern is at
> `tests/unit/engine/test_executors.py:6034-6043`. An existing
> in-memory metric reader fixture also lives at
> `tests/unit/telemetry/conftest.py:27` (`in_memory_metric_reader`) —
> the preferences-service variant below is the same idea scoped to the
> preferences module's meter and its module-import-time counter handle.
>
> **Why this is load-bearing for guard #3.** `_PREFERENCES_PATCH_COUNTER`
> is created at module-import time against `service._meter`. A test that
> only patches `service._meter` will not retroactively rebind the already-
> created counter handle; the counter will keep emitting to the old
> meter and the in-memory reader will see nothing. The fixture below
> rebinds BOTH `service._meter` AND `service._PREFERENCES_PATCH_COUNTER`
> so the counter `add(...)` calls land where the in-memory reader can
> see them, and restores both on teardown so other tests are not
> polluted.
>
> Extend `tests/unit/web/preferences/conftest.py` (create if absent):
>
> ```python
> # tests/unit/web/preferences/conftest.py
> from opentelemetry.sdk.metrics import MeterProvider
> from opentelemetry.sdk.metrics.export import InMemoryMetricReader
> import pytest
>
> from elspeth.web.preferences import service as _service_module
>
>
> @pytest.fixture
> def preferences_metric_reader(monkeypatch):
>     """Rebind service._meter + _PREFERENCES_PATCH_COUNTER to a fresh
>     MeterProvider with an InMemoryMetricReader attached.
>
>     Load-bearing for Task 3 guard #3 — the
>     ``_PREFERENCES_PATCH_COUNTER`` emit tests need to read counter
>     attributes without polluting global meter state. ``monkeypatch``
>     restores ``_meter`` and ``_PREFERENCES_PATCH_COUNTER`` on
>     teardown.
>
>     Reference pattern:
>     - ``tests/unit/engine/test_executors.py:6034-6043`` (canonical
>       MeterProvider + InMemoryMetricReader rebind).
>     - ``tests/unit/telemetry/conftest.py:27`` (existing
>       ``in_memory_metric_reader`` fixture for the telemetry module).
>     """
>     reader = InMemoryMetricReader()
>     provider = MeterProvider(metric_readers=[reader])
>     new_meter = provider.get_meter("preferences")
>     monkeypatch.setattr(_service_module, "_meter", new_meter)
>     # The counter handle was created at module-import time against the
>     # original meter. Rebind it against the new meter so emits land in
>     # the InMemoryMetricReader. Match the live description verbatim
>     # when you copy this into the test tree.
>     monkeypatch.setattr(
>         _service_module,
>         "_PREFERENCES_PATCH_COUNTER",
>         new_meter.create_counter(
>             name="composer.preferences.patch_total",
>             description=_service_module._PREFERENCES_PATCH_COUNTER.description,
>         ),
>     )
>     yield reader
> ```
>
> Define the helper alongside the new tests at the top of
> `tests/unit/web/preferences/test_service.py` (after imports, before
> the new test functions):
>
> ```python
> # tests/unit/web/preferences/test_service.py — module-level helper.
> from typing import Any
>
> from opentelemetry.sdk.metrics.export import InMemoryMetricReader
>
>
> def _last_patch_counter_attributes(
>     reader: InMemoryMetricReader,
> ) -> dict[str, Any]:
>     """Read ``composer.preferences.patch_total``'s latest data-point
>     attributes from the in-memory metric reader.
>
>     Load-bearing for Task 3 guard #3 (counter emit). Walks the
>     ``resource_metrics → scope_metrics → metrics`` structure produced
>     by ``InMemoryMetricReader.get_metrics_data()`` to locate the
>     preferences-patch counter, then returns the attributes dict from
>     the most-recently-recorded data point. Raises ``AssertionError``
>     if the counter has no data points — that condition indicates a
>     guard #3 regression (emit did not fire), and the named error
>     surfaces it cleanly rather than failing on a ``KeyError`` deeper
>     in the test.
>     """
>     data = reader.get_metrics_data()
>     for resource_metric in data.resource_metrics:
>         for scope_metric in resource_metric.scope_metrics:
>             for metric in scope_metric.metrics:
>                 if metric.name == "composer.preferences.patch_total":
>                     points = list(metric.data.data_points)
>                     if points:
>                         return dict(points[-1].attributes)
>     raise AssertionError(
>         "composer.preferences.patch_total had no data points — "
>         "counter emit did not fire (guard #3 regression).",
>     )
> ```
>
> The two new tests
> (`test_patch_tutorial_emits_counter_with_tutorial_changed_label`,
> `test_patch_without_tutorial_emits_counter_with_tutorial_changed_false`)
> below MUST take `preferences_metric_reader` as a fixture argument and
> call `_last_patch_counter_attributes(preferences_metric_reader)` —
> not the no-argument form. (The no-argument call sites in the original
> test sketch below are corrected to take the reader fixture.)

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


def test_corrupt_tutorial_completed_at_crashes_with_named_error(service, engine):
    """Tier-1 guard: a stored value that's neither NULL nor a datetime crashes
    with the *named* ``CorruptPreferencesError``, not bare ``RuntimeError``.

    Assertion is tightened to the named class so a future regression that
    substitutes bare ``RuntimeError`` is caught — ``CorruptPreferencesError``
    subclasses ``RuntimeError`` for forward-fit transition headroom, so
    ``pytest.raises(RuntimeError, …)`` would silently accept either form
    and let the regression land.
    """
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
    with pytest.raises(CorruptPreferencesError) as excinfo:
        asyncio.run(service.get_composer_preferences("alice-tutorial-corrupt"))
    # The named exception carries structured debug context.
    assert excinfo.value.user_id == "alice-tutorial-corrupt"
    assert excinfo.value.bad_value == {"tutorial_completed_at": "not-a-timestamp"}


def test_empty_patch_for_no_row_user_does_not_insert(service, engine):
    """Panel C2 preservation: empty PATCH against a no-row user is a no-op.

    Phase 1A's `update_composer_preferences` short-circuits before the
    INSERT...ON CONFLICT when (a) no row exists for the user AND (b) the
    payload sets no fields. The Phase 4 additive diff must NOT collapse
    this guard. If the regression returns, this test fails because a row
    will appear in `user_preferences_table` post-PATCH.
    """
    result = asyncio.run(
        service.update_composer_preferences(
            "alice-empty-patch", UpdateComposerPreferencesRequest()
        )
    )
    # Response is the lazy-default shape — no fabricated updated_at.
    assert result.default_mode == "guided"
    assert result.updated_at is None
    # And critically: no row was written.
    with engine.connect() as conn:
        count = conn.execute(
            select(user_preferences_table).where(
                user_preferences_table.c.user_id == "alice-empty-patch"
            )
        ).all()
    assert count == [], "Panel C2 guard regressed: empty PATCH inserted a row"


def test_patch_tutorial_emits_counter_with_tutorial_changed_label(
    service, preferences_metric_reader
):
    """Panel S1 preservation: `_PREFERENCES_PATCH_COUNTER` emits with the
    full label set including the new `tutorial_changed` label.

    Verifies both that the existing `wrote_row` label survived the additive
    diff AND that the new `tutorial_changed` label was added. Uses the
    ``preferences_metric_reader`` fixture (see conftest.py — Step 1
    prerequisite above) and the module-level
    ``_last_patch_counter_attributes`` helper.
    """
    stamp = datetime(2026, 5, 19, 12, 0, tzinfo=UTC)
    asyncio.run(
        service.update_composer_preferences(
            "alice-counter-tut",
            UpdateComposerPreferencesRequest(tutorial_completed_at=stamp),
        )
    )
    attrs = _last_patch_counter_attributes(preferences_metric_reader)
    assert attrs["mode_changed"] is False
    assert attrs["banner_dismissed"] is False
    assert attrs["wrote_row"] is True  # Phase 1A label preserved.
    assert attrs["tutorial_changed"] is True  # Phase 4 label extension.


def test_patch_without_tutorial_emits_counter_with_tutorial_changed_false(
    service, preferences_metric_reader
):
    """Counter-label disaggregation: a non-tutorial PATCH still emits the
    counter but with `tutorial_changed=False`."""
    asyncio.run(
        service.update_composer_preferences(
            "alice-counter-mode",
            UpdateComposerPreferencesRequest(default_mode="freeform"),
        )
    )
    attrs = _last_patch_counter_attributes(preferences_metric_reader)
    assert attrs["mode_changed"] is True
    assert attrs["wrote_row"] is True
    assert attrs["tutorial_changed"] is False  # absent from payload.
```

- [ ] **Step 2: Run to fail.** `.venv/bin/python -m pytest tests/unit/web/preferences/test_service.py -v` → FAIL (service does not yet read or write `tutorial_completed_at`).

- [ ] **Step 3: Extend the service — ADDITIVE DIFF, NOT REWRITE.**

Extend `_row_to_prefs()` and `update_composer_preferences()` in
`src/elspeth/web/preferences/service.py` to handle the new
`tutorial_completed_at` column. **This is an additive diff** — every
existing Phase 1A guard MUST be preserved. The full surrounding function
bodies are omitted here intentionally; consult the live file
(`src/elspeth/web/preferences/service.py`) before editing. The blocks below
show only the new logic for the new column plus the call-site for the
counter-label extension.

A first-pass plan review (2026-05-19, five reviewers converging:
reality:B-02, architecture:A3+A4, quality:Q2+Q3) flagged that a verbatim
full replacement of these functions silently dropped six Phase 1A
correctness mechanisms. **Do not regress them.** Before opening the file,
read this checklist and verify each item is still present in the live
file when you finish:

| # | Guard | Live `service.py` location | Failure mode if regressed |
|---|---|---|---|
| 1 | Panel C2 empty-PATCH-on-no-row guard | lines ~186, ~226-231 (`payload_is_empty` + the `if exists is None: return … False` short-circuit) | An empty PATCH against a never-seen user inserts a default row, contradicting the lazy-write contract on the GET side. |
| 2 | `wrote_row` bool tracking | `_sync` returns a 3-tuple ending in `wrote: bool`; consumed by the counter and the response `updated_at` | Counter cannot disaggregate "real write" from "no-op PATCH"; response carries fabricated `updated_at`. |
| 3 | `_PREFERENCES_PATCH_COUNTER` emit | lines ~84-88 (declare), lines ~293-300 (emit with `{mode_changed, banner_dismissed, wrote_row}` attributes) | Loss of operational-telemetry signal (Panel S1). |
| 4 | `CorruptPreferencesError(user_id, bad_value)` named raise | declared at lines ~47-69; raised at line ~151 (`_row_to_prefs`) and line ~259 (PATCH-side read of `existing_raw`) | Bare `RuntimeError` loses structured `user_id` + `bad_value` debug context; breaks Tier-1 read-guard parity between GET and PATCH paths. |
| 5 | Panel S3 `typing.cast()` pattern in `_row_to_prefs` | line ~252 (`insert_mode = cast(ComposerMode, existing_raw)` after the `existing_raw in _VALID_MODES` runtime check) | mypy stops catching `Any → ComposerMode` slips; the type contract becomes implicit/cargo-cult instead of visible. |
| 6 | Panel U1 lazy-default `updated_at=None` and Panel U1 corollary `updated_at=now if wrote else None` | lazy-default branch at lines ~128-132 (`updated_at=None`); response builder at line ~308 (`updated_at=now if wrote else None`) | Fabricates an `updated_at` timestamp for a row that was never written — fills an audit-visible field with a value the system never wrote. |

**Convention split (P1 contract, retained for the implementer's eyes
where the code lives):**

- `default_mode` and `banner_dismissed_at` keep the Phase 1A
  "**absent = preserve, None = collapse to no-op**" convention (one-way
  fields; no client need to NULL either via PATCH).
- `tutorial_completed_at` uses the **three-state discrimination via
  `model_fields_set`** (absent = preserve, datetime = set, null = clear) —
  required by Phase 8 Task 6's retake contract. See §"Cross-plan
  contract — `tutorial_completed_at` PATCH semantics" for the table.

Now the deltas.

**Delta A — extend `_row_to_prefs` with a Tier-1 read guard for the new
column.** Add after the existing mode-validation block; extend the
existing `return ComposerPreferences(...)` to include the new keyword
argument as shown below.

```python
# === Phase 4 additive delta in _row_to_prefs ===
# DO NOT modify the surrounding function structure.
# Preserved Phase 1A invariants (verify in live service.py before editing):
#   - Panel S3 cast() pattern on `existing_raw` mode-narrowing — NOT touched here
#     (that branch lives in update_composer_preferences, not _row_to_prefs).
#   - The existing `mode not in _VALID_MODES` raise of CorruptPreferencesError
#     stays as-is. We are appending a second, structurally parallel guard.
# Tier-1 guard: must be None or a datetime. SQLite's permissive type-affinity
# tolerates strings in a DateTime-typed column, so a corrupt row may surface
# here as a string or other type. Raise the NAMED exception with the offending
# value packed as a dict-keyed-by-column so an auditor reading the error can
# tell which column was corrupt without inspecting message text.
raw_tutorial = row.tutorial_completed_at
if raw_tutorial is None:
    tutorial_completed_at: datetime | None = None
elif isinstance(raw_tutorial, datetime):
    tutorial_completed_at = raw_tutorial
else:
    raise CorruptPreferencesError(
        user_id=user_id,
        bad_value={"tutorial_completed_at": raw_tutorial},
    )

return ComposerPreferences(
    default_mode=mode,
    banner_dismissed_at=row.banner_dismissed_at,
    tutorial_completed_at=tutorial_completed_at,  # NEW field on response model.
    updated_at=row.updated_at,
)
```

Note that `CorruptPreferencesError`'s `__init__` is the Phase 1A signature
`(user_id: str, bad_value: object)`. The Phase 4 use packs the bad value as
`{"tutorial_completed_at": raw_tutorial}` so the column-name is structurally
attached to the offending value rather than living only in the message
string. The existing `_row_to_prefs` mode-guard call site (line ~151) is
unchanged; do not retrofit a dict wrapper there.

**Delta B — lazy-default branch of `get_composer_preferences`.** Find the
`return ComposerPreferences(...)` in the no-row branch (live lines
~128-132). Add the new field with `tutorial_completed_at=None`. **Leave
`updated_at=None` exactly as-is** — Panel U1: no write event exists to
associate a timestamp with, so we do not fabricate one. The plan's
original verbatim rewrite incorrectly substituted `updated_at=self._now()`;
that would have written a synthetic timestamp into an audit-visible field.
Do not propagate that mistake into the live code.

```python
# === Phase 4 additive delta in lazy-default branch ===
# Existing fields unchanged; add `tutorial_completed_at=None`.
# Panel U1: updated_at stays None — no write event to attach a timestamp to.
return ComposerPreferences(
    default_mode=_DEFAULT_MODE,
    banner_dismissed_at=None,
    tutorial_completed_at=None,  # NEW.
    updated_at=None,  # PRESERVED — do NOT change to self._now().
)
```

**Delta C — extend `update_composer_preferences` to write
`tutorial_completed_at` using `model_fields_set` discrimination.** The
existing function body has a `_sync()` closure with three logical blocks:
(i) Panel C2 empty-PATCH-on-no-row guard, (ii) mode resolution (including
the `cast(ComposerMode, existing_raw)` Panel S3 line and the
`CorruptPreferencesError` raise on the PATCH-side read of `existing_raw`),
(iii) banner resolution. **All three blocks stay.** Add a fourth
resolution block and extend the `values` dict + `update_clause` dict.

The empty-PATCH predicate `payload_is_empty` (live line ~186) was
originally defined as `payload.default_mode is None and
payload.banner_dismissed_at is None`. Extend it to account for the new
field's three-state semantics — a `tutorial_completed_at` that is *present
in the payload* (whether as a datetime or as explicit null) is a real
edit and MUST disable the empty-PATCH short-circuit:

```python
# === Phase 4 additive delta — payload_is_empty predicate ===
# Preserve Phase 1A semantic: "default_mode and banner_dismissed_at use
# absent=preserve, None=collapse." Extend with the three-state field via
# `model_fields_set` — a key present in the payload (datetime OR explicit
# null) is a real edit.
tutorial_in_payload = "tutorial_completed_at" in payload.model_fields_set
payload_is_empty = (
    payload.default_mode is None
    and payload.banner_dismissed_at is None
    and not tutorial_in_payload
)
```

Inside `_sync()`, add the tutorial-resolution block AFTER the existing
banner-resolution block and BEFORE the `values` dict is built:

```python
# === Phase 4 additive delta — tutorial_completed_at resolution ===
# Three semantic states (see §"Cross-plan contract — `tutorial_completed_at`
# PATCH semantics"):
#   (a) field absent from payload         → preserve existing value
#   (b) field present, datetime value     → write the timestamp
#   (c) field present, value is None      → write NULL (Phase 8 retake)
# Pydantic v2 `model_fields_set` is the discriminator between (a) and (c);
# no sentinel needed.
if tutorial_in_payload:
    resolved_tutorial = payload.tutorial_completed_at  # datetime OR None.
else:
    resolved_tutorial = conn.execute(
        select(user_preferences_table.c.tutorial_completed_at).where(
            user_preferences_table.c.user_id == user_id
        )
    ).scalar_one_or_none()
```

Extend the `values` and `update_clause` dicts (live lines ~272-284). The
`values` dict gets the resolved tutorial value (insert-side); the
`update_clause` gets a *conditional* entry — only included when the
caller explicitly addressed the column — so the convention split is
faithful:

```python
# === Phase 4 additive delta — INSERT values ===
# Add tutorial_completed_at on insert-side: use the *resolved* value so a
# fresh-row upsert without the field in the payload writes NULL rather
# than the stale-read default. Existing keys unchanged.
values: dict[str, object] = {
    "user_id": user_id,
    "default_composer_mode": insert_mode,
    "banner_dismissed_at": payload.banner_dismissed_at,
    "tutorial_completed_at": resolved_tutorial,  # NEW.
    "updated_at": now,
}

# === Phase 4 additive delta — ON CONFLICT update_clause ===
# Only write the column on conflict if the caller explicitly addressed it.
# This is where the convention split lives: default_mode and
# banner_dismissed_at keep the Phase 1A `is not None` predicate;
# tutorial_completed_at uses the `model_fields_set` predicate so an
# explicit null reaches the write path. Existing two `if … is not None`
# blocks for default_mode and banner_dismissed_at are PRESERVED.
if tutorial_in_payload:
    # Writes either a datetime (set) or NULL (Phase 8 retake).
    update_clause["tutorial_completed_at"] = payload.tutorial_completed_at
```

**Delta D — preserve the `_sync` return shape AND add a fourth element
for the counter label.** Live `_sync` returns
`tuple[ComposerMode, datetime | None, bool]`. Extend to
`tuple[ComposerMode, datetime | None, datetime | None, bool, bool]` —
adding `resolved_tutorial` and the new `tutorial_in_payload` flag. The
final `bool` is still `wrote_row` (guard #2 — do not remove). The new
`tutorial_in_payload` flag flows out as a counter label, NOT a response
field, because the response uses `resolved_tutorial` directly:

```python
# === Phase 4 additive delta — _sync return tuple ===
# Existing shape: (insert_mode, resolved_banner, wrote: bool)
# New shape:     (insert_mode, resolved_banner, resolved_tutorial,
#                  tutorial_changed: bool, wrote: bool)
# `tutorial_changed` mirrors the existing mode_changed/banner_dismissed
# attribute style on the counter, computed as `tutorial_in_payload`
# (whether the caller explicitly addressed the column).
return insert_mode, resolved_banner, resolved_tutorial, tutorial_in_payload, True

# The Panel C2 short-circuit branch (no-row + empty PATCH) returns:
return _DEFAULT_MODE, None, None, False, False
```

**Delta E — counter-label extension at the post-`_sync` emit (live lines
~293-300). DO NOT REWRITE the counter declaration or the emit-site
structure.** Add `tutorial_changed` to the `attributes` dict. The
declaration's docstring (line ~87) should also gain the new label name
in the same edit for documentation parity:

```python
# === Phase 4 additive delta — counter declaration docstring (line ~87) ===
# Existing: "Attributes: mode_changed (bool), banner_dismissed (bool),
#            wrote_row (bool)."
# New:      "Attributes: mode_changed (bool), banner_dismissed (bool),
#            wrote_row (bool), tutorial_changed (bool)."

# === Phase 4 additive delta — counter emit (lines ~293-300) ===
# PRESERVE the existing _PREFERENCES_PATCH_COUNTER.add(1, attributes={...})
# call structure. Add the new `tutorial_changed` label only.
written_mode, written_banner, written_tutorial, tutorial_changed, wrote = (
    await run_sync_in_worker(_sync)
)
_PREFERENCES_PATCH_COUNTER.add(
    1,
    attributes={
        "mode_changed": payload.default_mode is not None,
        "banner_dismissed": payload.banner_dismissed_at is not None,
        "wrote_row": wrote,                  # PRESERVED — guard #2.
        "tutorial_changed": tutorial_changed,  # NEW — Phase 4 label.
    },
)
return ComposerPreferences(
    default_mode=written_mode,
    banner_dismissed_at=written_banner,
    tutorial_completed_at=written_tutorial,  # NEW response field.
    # PRESERVED — Panel U1 corollary tied to guard #2:
    #   wrote=False means the Panel C2 short-circuit fired; no write event
    #   exists, so do not fabricate `updated_at=now`.
    updated_at=now if wrote else None,
)
```

**Do NOT add `slog.debug(...)` calls anywhere in this code.** The counter
emit is the operational-telemetry signal; logging would violate the
audit/telemetry/logging primacy ordering (see CLAUDE.md and memory
`feedback_no_slog_recommendations.md`).

**Imports.** `datetime` and `cast` are already imported in the live file
(lines 30-31); no new imports needed. `CorruptPreferencesError` is defined
locally in the same module (line ~47); referenced by name.

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
    """Corruption guard: a present-but-unparseable file is a fault, not a miss."""
    # Compute the key the way the cache does (this mirrors the internal hash).
    from elspeth.web.preferences.tutorial_cache import _compute_key
    key = _compute_key(CANONICAL_SEED_PROMPT, "claude-opus-4-7")
    (cache_dir / f"{key}.json").write_text("this is not json")
    with pytest.raises(RuntimeError, match="tutorial cache"):
        cache.lookup(CANONICAL_SEED_PROMPT, "claude-opus-4-7")


def test_file_with_mismatched_prompt_crashes_lookup(
    cache: TutorialCache, cache_dir: Path
) -> None:
    """Corruption guard: an in-place file whose contents disagree with the key.

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

## Task 7.0: Schema — extend `runs_table` with audit-story columns

**Files:**
- Modify: `src/elspeth/core/landscape/schema.py` — add six columns to `runs_table`.
- Modify: the Landscape write path that records a completed run (recon below) — populate the new columns.
- Create: `tests/unit/core/landscape/test_runs_table_audit_story_columns.py` — column-presence + write-side fixture tests.

**Why this is a separate task (and why it lands before Task 7).** The
audit-story endpoint added in Task 7.2 reads six fields from a Landscape
run row: `llm_call_count`, `source_data_hash`, `run_started_at`,
`plugin_versions`, `seeded_from_cache`, `cache_key`. Verified against the
live schema (`src/elspeth/core/landscape/schema.py:61-98` as of
2026-05-19), **none of these columns exist on `runs_table` today**. The
table has `started_at` (close to but not `run_started_at`), `config_hash`,
and other run-config metadata, but no per-run output-hash, no LLM-call
counter, no plugin-version manifest, no cache-replay markers.

Task 7's replay path (`_replay_cached_content_to_landscape`) and Task 7.2's
read path both assume these columns. Without Task 7.0, Task 7 will write
metadata that has no column to land in, and Task 7.2's
`audit_row.llm_call_count` access will raise `AttributeError` at runtime.

**Operator action required.** Per the project's no-Alembic policy (memory:
`project_db_migration_policy`, `project_phase9_sqlite_only`), schema
changes are applied by deleting the existing audit DB and letting the
caretaker re-bootstrap from `schema.py`. **The operator must delete
`/var/lib/elspeth/landscape/audit.db` (and any per-eval audit DBs) after
this task lands, before the next pipeline run.** This is the established
DB-migration pattern, but its scope here is the full Landscape audit DB —
surface this to the operator and pause for confirmation before merging
Task 7.0.

**Trust tier.** All six new columns are Tier-1 (Landscape data). Their
write-side population (in Task 7's modified run path and in the
existing run-completion path) must be unconditional — there is no
"populate if available" branch. If the run path cannot supply one of these
fields, the run is corrupt and must fail at the write site, not silently
write NULL into a column the audit-story endpoint expects to find populated.

- [ ] **Step 1: Reconnaissance — identify the live Landscape run-completion write site.**

```bash
grep -rn "runs_table.*insert\|INSERT.*INTO.*runs\|record_run_completion\|persist_run\b" \
  src/elspeth/core/landscape/ --include="*.py" | grep -v __pycache__ | head -20
```

Identify the function that inserts a row into `runs_table` at the end of a
successful pipeline run. Confirm the shape of the data it receives:

1. Where does `llm_call_count` come from? Likely a counter accumulated by
   the LLM transform's metrics — find the accumulator and the read site.
2. Where does `source_data_hash` come from? Likely computed by the source
   loader (the `source_data_hash` already referenced in
   `_replay_cached_content_to_landscape`). Confirm name match.
3. `run_started_at` — the run-orchestrator already records this; it may be
   `started_at` already. If so, this column is a rename, not an add.
4. `plugin_versions` — the bootstrap path resolves plugin versions; find
   the in-memory representation and the serialization point.
5. `seeded_from_cache` / `cache_key` — these are written ONLY by Task 7's
   `_replay_cached_content_to_landscape`; on a normal live run they are
   `False` / `None`. Confirm with the operator whether the design treats
   them as nullable-but-Tier-1 (the live-run row writes `seeded_from_cache
   = False` and `cache_key = NULL` explicitly) or as cache-only metadata
   stored in a side table — the simpler design is the former.

Write the findings into the commit body. If any of the above does not
exist in the live codebase (e.g., LLM call counter is not currently
accumulated at run scope), surface that to the operator before proceeding;
Task 7.0 may need to expand to add the accumulation, not just the storage.

- [ ] **Step 2: Write the failing test.**

Create `tests/unit/core/landscape/test_runs_table_audit_story_columns.py`:

```python
"""Column-presence and write-side smoke tests for Task 7.0 additions."""

from __future__ import annotations

from elspeth.core.landscape.schema import runs_table


def test_runs_table_has_audit_story_columns() -> None:
    """The six Phase 4.7.2 audit-story columns are present on runs_table."""
    expected = {
        "llm_call_count",
        "source_data_hash",
        "run_started_at",
        "plugin_versions",
        "seeded_from_cache",
        "cache_key",
    }
    actual = {col.name for col in runs_table.columns}
    missing = expected - actual
    assert not missing, f"Missing columns on runs_table: {missing}"


def test_llm_call_count_is_non_nullable_integer() -> None:
    """Tier-1 invariant: every run has a definite LLM call count (0 is legal)."""
    col = runs_table.c.llm_call_count
    assert not col.nullable, "llm_call_count must be NOT NULL (Tier-1)"
    # Python type: int; SQL type left to the test author's verification.
```

Expect: FAIL — columns do not exist yet.

- [ ] **Step 3: Add the columns.**

In `src/elspeth/core/landscape/schema.py`, extend `runs_table` (the existing
`Table("runs", ...)` block at lines 61-98) with the six new columns. Place
them after `runtime_val_manifest_json` (current last column at line 97) so
the column-order on existing rows is preserved (relevant for the
delete-and-recreate migration path):

```python
# === Phase 4A.7.0 audit-story columns (added 2026-05-19) ===
# These six columns back the GET /audit-story endpoint (Phase 4A.7.2).
# All Tier-1: write-side MUST populate every column on every run.
# `seeded_from_cache` and `cache_key` default to False / NULL for live runs
# and are populated only by `_replay_cached_content_to_landscape` (Task 7).
Column("llm_call_count", Integer, nullable=False),
Column("source_data_hash", String(64), nullable=False),
# Renamed from / aliased to `started_at` if recon confirms the live column.
# If recon shows `started_at` is the canonical run-start timestamp, DO NOT
# add a duplicate column — instead update Task 7.2's service to read
# `started_at` and rename the response field. Document the decision.
Column("run_started_at", DateTime(timezone=True), nullable=False),
Column("plugin_versions", Text, nullable=False),  # canonical-JSON dict[str,str]
Column("seeded_from_cache", Boolean, nullable=False, default=False),
Column("cache_key", String(64), nullable=True),
```

Note: `Integer`, `Boolean`, and `Text` must be imported from `sqlalchemy`
alongside the existing `Column`, `String`, `DateTime` imports.

- [ ] **Step 4: Extend the run-completion write site.**

In the file Step 1 identified, extend the `runs_table` INSERT to populate
the six new columns. For live runs:

- `llm_call_count`: read from the run-scope LLM counter (Step 1 recon).
- `source_data_hash`: read from the source-loader output (Step 1 recon).
- `run_started_at`: same value the existing `started_at` column receives
  (if recon confirms `started_at` is the run-start timestamp, see Step 3
  note).
- `plugin_versions`: serialise the bootstrap-resolved plugin versions as
  canonical JSON.
- `seeded_from_cache`: `False`.
- `cache_key`: `None`.

For cache-replay runs (the path in Task 7's
`_replay_cached_content_to_landscape`), the values come from the cached
entry plus the replay's own cache-key computation. Task 7's
implementation must be updated to pass these through to the
run-completion write site (or the replay path may write directly to
`runs_table` if it bypasses the standard write — confirm during Step 1).

- [ ] **Step 5: Run the column-presence test to verify it passes.**

```bash
.venv/bin/python -m pytest tests/unit/core/landscape/test_runs_table_audit_story_columns.py -v
```

Expected: PASS.

- [ ] **Step 6: Operator gate — DB delete.**

```text
Operator action: Delete the live Landscape audit DBs before the next
pipeline run.

Paths to remove (verify with the operator — exact paths depend on the
deployment):
  /var/lib/elspeth/landscape/audit.db        (production)
  examples/*/runs/audit.db                   (example pipelines)
  evals/*/audit.db                           (eval harness)

Confirm with operator. Do NOT proceed to Task 7 until this is acknowledged.
```

- [ ] **Step 7: Commit.**

```bash
git add src/elspeth/core/landscape/schema.py \
        <run-completion-write-site> \
        tests/unit/core/landscape/test_runs_table_audit_story_columns.py
git commit -m "feat(landscape): add audit-story columns to runs_table (Phase 4A.7.0)"
```

---

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
    """Corruption guarantee: corrupt cache must crash (500), not silently bypass.

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

## Task 7.1: Route + service — `POST /api/tutorial/run`

**Files:**

- Create: `src/elspeth/web/composer/tutorial_run_routes.py` — FastAPI router.
- Create: `src/elspeth/web/composer/tutorial_service.py` — service method.
- Create: `tests/integration/web/test_tutorial_routes.py` — integration tests.
- Modify: `src/elspeth/web/composer/__init__.py` — export the router.
- Modify: the FastAPI app-composition site (identified in Task 6 recon) — `include_router(create_tutorial_run_router())`.

Task 7 wires the cache-consult branch into the *generic* run path (so any
canonical-seed run benefits, including ones initiated from the existing
`POST /api/sessions/{id}/runs` endpoint). Task 7.1 adds the **tutorial-specific
entry point** that the frontend's `runTutorialPipeline` (21b2 Task 8) calls.
The route is a thin façade over the run-path orchestration Task 7 already
wired: it accepts a `(session_id, prompt)` pair, derives the canonical
pipeline from the prompt, invokes the run-path, and returns the
`TutorialRunResponse` shape the frontend consumes.

The route does **not** duplicate Task 7's cache logic — it calls a service
method (`run_tutorial_pipeline`) which delegates to the same
`execute_pipeline_run` (or recon-confirmed equivalent) entry point that
Task 7 modified. The cache-consult branch fires inside that entry point;
Task 7.1 simply guarantees the frontend has a stable, narrow surface.

**Bypass paths (live run, no cache consult, no cache write):**

1. User's `tutorial_completed_at IS NOT NULL` — post-completion users get
   live runs (they're past the tutorial; cache hits would be misleading).
2. User's `default_mode == 'freeform'` — freeform users skipped the tutorial
   by choice; the cache is a tutorial-only optimisation (per Q11). A
   freeform user who *does* invoke `POST /api/tutorial/run` (e.g. via the
   Phase 8 retake button) gets a live run.

Both bypass paths are evaluated **before** any cache lookup. The bypass
decision is logged to the Landscape entry's metadata as
`tutorial_cache_bypass_reason: "completed" | "freeform" | None` so an
auditor can later distinguish a bypass from a miss.

- [ ] **Step 1: Reconnaissance — confirm the run-path service surface.**

```bash
grep -n "def execute_pipeline_run\|def _execute_pipeline_live\|create_run_router" \
  src/elspeth/web/sessions/*.py src/elspeth/web/composer/*.py 2>/dev/null
```

Confirm:

1. The exact name of the function Task 7 modified (its argument shape
   determines `run_tutorial_pipeline`'s pass-through call).
2. The auth dep used by sibling composer routes (`get_current_user` or a
   composer-specific dep) — match it.
3. The existing FastAPI app-composition site where the new router is
   `include_router`'d — same file as `create_preferences_router()` is
   registered.
4. The Pydantic config base (project convention is `ConfigDict(strict=True,
   extra='forbid')` — confirm against an existing model such as
   `UpdateSessionRequest` in `src/elspeth/web/sessions/schemas.py`).
5. **Session-ownership verification is a shared free function**:
   `verify_session_ownership(session_id, user, request)` in
   `src/elspeth/web/sessions/ownership.py`. It raises
   `HTTPException(404)` on any access-control failure (unknown session,
   wrong user, wrong auth provider) — **the IDOR contract is 404, not
   403**, deliberately, to avoid leaking session existence to a UUID
   enumerator. Do NOT reinvent this check inline; do NOT add a
   service-method shim such as `session_exists_for_user`. Reuse the
   free function. Confirm by reading `sessions/ownership.py:26-50`.

Write the findings into the commit message body. Do not guess.

- [ ] **Step 2: Write the failing integration test.**

Create `tests/integration/web/test_tutorial_routes.py`:

```python
"""Integration tests for POST /api/tutorial/run."""

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
from elspeth.web.composer.tutorial_run_routes import create_tutorial_run_router
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
def app(cache_dir: Path) -> FastAPI:
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
    app.include_router(create_tutorial_run_router())
    return app


def _seed_canonical_cache_entry(app: FastAPI) -> None:
    app.state.tutorial_cache.store(
        TutorialCacheEntry(
            canonical_prompt=CANONICAL_SEED_PROMPT,
            model_id="claude-opus-4-7",
            cached_at=datetime(2026, 5, 15, tzinfo=UTC),
            rows=[{"url": "ato.gov.au", "score": 5, "rationale": "clear"}],
            source_data_hash="a7f3e2cached",
            llm_call_count=5,
            pipeline_yaml="<canonical>",
        )
    )


def _create_session(app: FastAPI, user_id: str = "alice") -> str:
    """Create a session row and return its id.

    Reuses the project's established integration-test fixture pattern from
    ``tests/integration/web/conftest.py`` (see ``_make_session``): the
    fixture inserts directly into ``sessions_table`` via the session
    engine. There is no ``create_session_for_user`` free function — the
    public production surface is the ``SessionsServiceImpl.create_session``
    coroutine, but tests bypass it to set up arbitrary ownership.
    Implementer: import ``_make_session`` from the existing conftest, or
    inline the same INSERT pattern here. Confirm during Step 1 recon.
    """
    raise NotImplementedError("use _make_session from tests/integration/web/conftest.py")


def test_post_run_cache_hit_returns_current_session_run_id(
    app: FastAPI, cache_dir: Path
) -> None:
    """Cache hit: response.run_id is owned by the current session; seeded_from_cache=True."""
    _seed_canonical_cache_entry(app)
    session_id = _create_session(app)
    client = TestClient(app)

    response = client.post(
        "/api/tutorial/run",
        json={"session_id": session_id, "prompt": CANONICAL_SEED_PROMPT},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["seeded_from_cache"] is True
    assert isinstance(body["cache_key"], str) and len(body["cache_key"]) == 64
    # The run_id is a fresh identifier owned by the caller's session
    # (cross-validated by querying the session's runs).
    assert isinstance(body["run_id"], str) and len(body["run_id"]) > 0
    # The output's source_data_hash matches the cached entry — content,
    # not identity, was replayed.
    assert body["output"]["source_data_hash"] == "a7f3e2cached"


def test_post_run_cache_miss_executes_fresh(
    app: FastAPI, cache_dir: Path
) -> None:
    """Cache miss: live run, response.seeded_from_cache=False, cache populated post-run."""
    session_id = _create_session(app)
    client = TestClient(app)

    response = client.post(
        "/api/tutorial/run",
        json={"session_id": session_id, "prompt": CANONICAL_SEED_PROMPT},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["seeded_from_cache"] is False
    assert body["cache_key"] is None
    # The cache was written post-run (one file in the cache directory).
    assert len(list(cache_dir.iterdir())) == 1


def test_post_run_bypasses_cache_for_freeform_user(
    app: FastAPI, cache_dir: Path
) -> None:
    """default_mode=='freeform' → cache bypassed even if a hit would be available."""
    _seed_canonical_cache_entry(app)
    session_id = _create_session(app)
    client = TestClient(app)
    # Set the user's mode to freeform.
    client.patch("/api/composer-preferences", json={"default_mode": "freeform"})

    # Snapshot cache directory before run; after the bypass the directory
    # must still hold ONLY the pre-seeded canonical entry (no new file).
    pre_run_files = {p.name for p in cache_dir.iterdir()}
    with patch(
        "elspeth.web.composer.tutorial_service.TutorialCache.lookup"
    ) as mock_lookup:
        response = client.post(
            "/api/tutorial/run",
            json={"session_id": session_id, "prompt": CANONICAL_SEED_PROMPT},
        )
        assert response.status_code == 200
        # The cache lookup was bypassed — never called.
        mock_lookup.assert_not_called()
    body = response.json()
    assert body["seeded_from_cache"] is False
    # Bypass contract: no cache write either (otherwise a bypass-mode run could
    # poison the cache for non-bypass users).
    post_run_files = {p.name for p in cache_dir.iterdir()}
    assert post_run_files == pre_run_files, (
        f"Bypass path must not write the cache; "
        f"new files: {post_run_files - pre_run_files}"
    )


def test_post_run_bypasses_cache_for_completed_user(
    app: FastAPI, cache_dir: Path
) -> None:
    """tutorial_completed_at IS NOT NULL → cache bypassed (post-completion live runs)."""
    _seed_canonical_cache_entry(app)
    session_id = _create_session(app)
    client = TestClient(app)
    client.patch(
        "/api/composer-preferences",
        json={"tutorial_completed_at": "2026-05-14T00:00:00Z"},
    )

    pre_run_files = {p.name for p in cache_dir.iterdir()}
    with patch(
        "elspeth.web.composer.tutorial_service.TutorialCache.lookup"
    ) as mock_lookup:
        response = client.post(
            "/api/tutorial/run",
            json={"session_id": session_id, "prompt": CANONICAL_SEED_PROMPT},
        )
        assert response.status_code == 200
        mock_lookup.assert_not_called()
    body = response.json()
    assert body["seeded_from_cache"] is False
    # Bypass contract: no cache write either (otherwise a bypass-mode run could
    # poison the cache for non-bypass users).
    post_run_files = {p.name for p in cache_dir.iterdir()}
    assert post_run_files == pre_run_files, (
        f"Bypass path must not write the cache; "
        f"new files: {post_run_files - pre_run_files}"
    )


def test_post_run_unknown_session_returns_404(
    app: FastAPI, cache_dir: Path
) -> None:
    """Unknown session → 404, raised by the shared ownership helper.

    Per the live IDOR contract (src/elspeth/web/sessions/ownership.py:33),
    verify_session_ownership raises HTTPException(404) for any
    access-control failure — unknown session, wrong user, or wrong auth
    provider. The 404 is deliberate (not 403) to avoid leaking session
    existence to an attacker enumerating UUIDs.
    """
    client = TestClient(app)
    response = client.post(
        "/api/tutorial/run",
        json={
            "session_id": "00000000-0000-0000-0000-000000000000",
            "prompt": CANONICAL_SEED_PROMPT,
        },
    )
    assert response.status_code == 404


def test_post_run_corrupt_preferences_returns_500(
    app: FastAPI, cache_dir: Path
) -> None:
    """Tier-1: corrupt preferences row → CorruptPreferencesError → 500."""
    session_id = _create_session(app)
    # Write a corrupt tutorial_completed_at directly via the engine (bypassing Pydantic).
    from sqlalchemy import text
    with app.state.session_engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO user_preferences_table (user_id, tutorial_completed_at) "
                "VALUES ('alice', 'not-a-datetime')"
            )
        )
    client = TestClient(app)
    response = client.post(
        "/api/tutorial/run",
        json={"session_id": session_id, "prompt": CANONICAL_SEED_PROMPT},
    )
    assert response.status_code == 500


def test_post_run_request_extra_field_rejected(
    app: FastAPI, cache_dir: Path
) -> None:
    """ConfigDict(extra='forbid') invariant: unknown fields in the body → 422."""
    session_id = _create_session(app)
    client = TestClient(app)
    response = client.post(
        "/api/tutorial/run",
        json={
            "session_id": session_id,
            "prompt": CANONICAL_SEED_PROMPT,
            "rogue_field": "should-reject",
        },
    )
    assert response.status_code == 422
```

Placeholders (`_create_session`, the corrupt-row INSERT shape) are filled in
from Step 1's recon; do not guess. The test file may not be valid Python
until then — intentional: TDD must fail against real code, not mocks.

- [ ] **Step 3: Run test to verify it fails.**

```bash
.venv/bin/python -m pytest tests/integration/web/test_tutorial_routes.py -v
```

Expected: FAIL — module-import error (router not yet created) or
behavioural failure.

- [ ] **Step 4: Implement the Pydantic models, route, and service.**

Create `src/elspeth/web/composer/tutorial_run_routes.py`:

```python
"""Tutorial run endpoint — POST /api/tutorial/run.

Phase 4A.7.1. The frontend's runTutorialPipeline (21b2 Task 8) calls this
route. Logic is delegated to TutorialRunService; the route is a thin
FastAPI shell that handles request parsing, auth, and response shaping.

Per CLAUDE.md no-defensive-programming: no try/except wrapping. Errors
from the service propagate to FastAPI's exception handlers (CorruptPreferencesError
maps to 500 via the global handler installed in app composition; HTTPException
short-circuits FastAPI; everything else surfaces as the framework default 500).
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, ConfigDict

from elspeth.web.auth.middleware import get_current_user
from elspeth.web.auth.models import UserIdentity


class TutorialRunRequest(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    session_id: str
    prompt: str


class TutorialRunOutput(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    rows: list[dict[str, Any]]
    source_data_hash: str


class TutorialRunResponse(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    run_id: str
    output: TutorialRunOutput
    seeded_from_cache: bool
    cache_key: str | None


def create_tutorial_run_router() -> APIRouter:
    router = APIRouter(prefix="/api/tutorial", tags=["tutorial"])

    @router.post("/run", response_model=TutorialRunResponse)
    async def run_tutorial(
        request: Request,
        body: TutorialRunRequest,
        user: UserIdentity = Depends(get_current_user),
    ) -> TutorialRunResponse:
        from elspeth.web.composer.tutorial_service import run_tutorial_pipeline

        # No try/except wrapping: the service calls verify_session_ownership
        # which raises HTTPException(404) directly on any access-control
        # failure (the IDOR contract; see
        # src/elspeth/web/sessions/ownership.py:33). HTTPException
        # short-circuits FastAPI's response pipeline; CorruptPreferencesError
        # propagates to the global 500 handler installed in app composition.
        return await run_tutorial_pipeline(
            request=request,
            user=user,
            session_id=body.session_id,
            prompt=body.prompt,
        )

    return router
```

Create `src/elspeth/web/composer/tutorial_service.py`:

```python
"""Tutorial run service — orchestrates cache consult, bypass, and live run.

Phase 4A.7.1. Layered above Task 7's `execute_pipeline_run` so the
cache-consult branch lives in exactly one place (Task 7's modified run
path). This service adds the **bypass** logic (completed-user, freeform-user)
that is specific to the tutorial entry point.

Tier model:
- session_id, prompt: Tier 3 (untrusted) — the composer LLM `set_pipeline`
  path already handles Tier 3 prompts; this service forwards.
- preferences row: Tier 1 — `CorruptPreferencesError` propagates.
- cache content: server-generated — corruption crashes (Task 5).
"""

from __future__ import annotations

from uuid import UUID

from fastapi import Request

from elspeth.web.auth.models import UserIdentity
from elspeth.web.composer.tutorial_run_routes import (
    TutorialRunOutput,
    TutorialRunResponse,
)
from elspeth.web.sessions.ownership import verify_session_ownership


async def run_tutorial_pipeline(
    *,
    request: Request,
    user: UserIdentity,
    session_id: str,
    prompt: str,
) -> TutorialRunResponse:
    """Execute the tutorial pipeline for ``user`` against ``session_id``.

    Decision order:

    1. Validate session ownership via the shared
       ``verify_session_ownership`` free function. Per the live IDOR
       contract (``src/elspeth/web/sessions/ownership.py:33``), any
       access-control failure — unknown session, wrong user, wrong auth
       provider — raises ``HTTPException(404)`` directly. The 404 (not
       403) is deliberate: distinguishing "no such session" from "you
       can't access this session" would leak session existence to a UUID
       enumerator. This service does NOT catch and re-raise; the
       HTTPException propagates to FastAPI's exception handlers.
    2. Read the user's preferences. Tier-1 corruption → CorruptPreferencesError
       (propagates to the 500 handler).
    3. Compute bypass reason:
       - prefs.tutorial_completed_at IS NOT NULL  → bypass ("completed")
       - prefs.default_mode == "freeform"         → bypass ("freeform")
       - otherwise                                → consult cache
    4. Build the canonical pipeline from ``prompt`` (the composer LLM
       set_pipeline path handles Tier-3 prompts; result is a pipeline_state
       dict matching Task 7's `_is_canonical_seed_pipeline` contract).
    5. Invoke Task 7's `execute_pipeline_run` — that function already
       contains the cache-consult / cache-store branches gated on
       prefs.tutorial_completed_at being None. On bypass paths we pass a
       pre-built pipeline_state whose mode-flag forces the live path.

    Per CLAUDE.md offensive-programming:
    - No `.get()` / `getattr(default)` against pipeline_state or prefs.
    - Bypass-reason is computed offensively; an unrecognised mode crashes
      (the existing `_VALID_MODES` guard in preferences/service.py catches
      this upstream).
    """
    # Reuse the shared IDOR-safe ownership check. Raises HTTPException(404)
    # on mismatch (no separate SessionNotFoundError indirection needed —
    # the shared helper already raises the framework-native error).
    await verify_session_ownership(
        session_id=UUID(session_id),
        user=user,
        request=request,
    )

    prefs_service = request.app.state.preferences_service
    prefs = await prefs_service.get_composer_preferences(user.user_id)

    bypass_reason: str | None = None
    if prefs.tutorial_completed_at is not None:
        bypass_reason = "completed"
    elif prefs.default_mode == "freeform":
        bypass_reason = "freeform"

    pipeline_state = _build_canonical_pipeline_from_prompt(
        prompt=prompt,
        force_live=bypass_reason is not None,
    )

    # Task 7's execute_pipeline_run owns the cache-consult/cache-store
    # branches. When force_live is set, _is_canonical_seed_pipeline returns
    # False (the pipeline_state carries a flag the helper checks) and the
    # cache is bypassed.
    from elspeth.web.composer.run_path import execute_pipeline_run  # recon-confirmed

    result = await execute_pipeline_run(
        request=request,
        user=user,
        session_id=session_id,
        pipeline_state=pipeline_state,
    )

    # Tier-1: every field below is read from a Landscape entry the run-path
    # just wrote (or replayed). Absence is corruption.
    return TutorialRunResponse(
        run_id=result["run_id"],
        output=TutorialRunOutput(
            rows=result["rows"],
            source_data_hash=result["source_data_hash"],
        ),
        seeded_from_cache=result["seeded_from_cache"],
        cache_key=result["cache_key"],
    )


def _build_canonical_pipeline_from_prompt(
    *, prompt: str, force_live: bool
) -> dict[str, object]:
    """Build a pipeline_state dict from the (possibly edited) seed prompt.

    Exact construction depends on Step 1 recon — likely a thin wrapper
    around the existing composer set_pipeline path. The `force_live` flag
    is attached as `pipeline_state["_tutorial_force_live"] = True`; Task 7's
    `_is_canonical_seed_pipeline` reads this flag and returns False when set
    (which bypasses both the cache-consult and the cache-store branches).
    """
    raise NotImplementedError("filled in from Step 1 recon")
```

Modify the FastAPI app-composition site (identified in Step 1 recon) to:

```python
from elspeth.web.composer.tutorial_run_routes import create_tutorial_run_router
app.include_router(create_tutorial_run_router())
```

- [ ] **Step 5: Update Task 7's `_is_canonical_seed_pipeline` to honour the force-live flag.**

In the run-path file Task 7 modified, extend `_is_canonical_seed_pipeline`:

```python
def _is_canonical_seed_pipeline(pipeline_state: Mapping[str, object]) -> bool:
    # Force-live escape hatch (Task 7.1): bypass paths short-circuit the canonical
    # check so cache_consult AND cache_write are both disabled. Use explicit
    # membership + identity check rather than `.get()` to keep the
    # optional-key contract visible (per CLAUDE.md offensive-programming —
    # `.get(...)` is the defensive-default form we avoid in production code).
    if "_tutorial_force_live" in pipeline_state and pipeline_state["_tutorial_force_live"] is True:
        return False
    # ... existing canonical-seed shape checks (recon) ...
```

This is the *only* call-site change Task 7.1 makes outside the new files.

- [ ] **Step 6: Run test to verify it passes.**

```bash
.venv/bin/python -m pytest tests/integration/web/test_tutorial_routes.py -v
```

Expected: PASS — all seven tests green.

- [ ] **Step 7: Run the full integration suite.**

```bash
.venv/bin/python -m pytest tests/integration/web/ -v
```

Expected: PASS — Task 7's `test_non_tutorial_user_skips_cache` and
`test_edited_prompt_skips_cache` still green; the new bypass paths do not
collide with their gates.

- [ ] **Step 8: Commit.**

```bash
git add src/elspeth/web/composer/tutorial_run_routes.py \
        src/elspeth/web/composer/tutorial_service.py \
        src/elspeth/web/composer/__init__.py \
        tests/integration/web/test_tutorial_routes.py \
        <app-composition-site> <run-path-file>
git commit -m "feat(web): POST /api/tutorial/run with completed/freeform bypass (Phase 4A.7.1)"
```

---

## Task 7.2: Route + service — `GET /api/sessions/{session_id}/runs/{run_id}/audit-story`

**Files:**

- Create: `src/elspeth/web/sessions/audit_story_service.py` — service method.
- Modify: `src/elspeth/web/sessions/routes.py` — add the new route handler.
- Modify: `src/elspeth/web/sessions/schemas.py` — add `RunAuditStoryResponse`.
- Create: `tests/integration/web/test_audit_story_routes.py` — integration tests.

This endpoint surfaces the **real Landscape audit row** for a specific
`(session_id, run_id)` pair so the frontend's Turn 5 (21b2 Task 9) can
render a load-bearing audit narrative against the user's own run. The
response is derived **entirely from real audit data**. **No field is ever
synthesised, defaulted, or inferred** — if a field is absent from the
audit row, that absence is a Tier-1 corruption signal and the service
raises `CorruptAuditRowError`, which propagates to 500. This is the Q6
no-synthesis invariant: the audit-story endpoint must not lie about what
the audit trail recorded, even by omission.

**Authorization order:**

1. Caller authenticated (route dep).
2. Caller owns `session_id` (session-ownership check, via
   `verify_session_ownership` in `src/elspeth/web/sessions/ownership.py`).
   If not → **404**. The IDOR contract (`sessions/ownership.py:33`) is 404
   on every access-control failure (unknown session, wrong user, wrong
   auth provider) — deliberately, to avoid leaking session existence to a
   UUID enumerator. Do **not** return 403 here; that would expose "this
   session exists, you just can't read it" vs "no such session". 403 is
   reserved for the unauthenticated case (no session at all), which is
   handled by the auth middleware before this route runs.
3. `run_id` belongs to `session_id` (cross-ownership query). If not → 404
   (an unknown `run_id` and a foreign-but-existing `run_id` are
   indistinguishable to the caller, by design). Same-user but
   cross-session also returns 404 (run not found in this session).
4. Tier-1 Landscape read. Missing required field → CorruptAuditRowError → 500.

- [ ] **Step 1: Reconnaissance — confirm the Landscape read surface.**

```bash
grep -n "def get_run_audit\|def read_run_audit\|landscape\.read\|run_id.*session_id" \
  src/elspeth/web/sessions/*.py src/elspeth/core/landscape/*.py 2>/dev/null | head -30
```

Confirm:

1. **The Landscape data is reached via TWO reads composed.** There is no
   `app.state.landscape.read_run(session_id, run_id)` shortcut and no
   `app.state.landscape` attribute on the app state. The composition is:
   - (a) **Composer-DB read** via the per-request session-service:
     `record = await session_service.get_run(UUID(run_id))` returns a
     `RunRecord` (`src/elspeth/web/sessions/protocol.py:430`). Use
     `record.session_id` to enforce the cross-session check (a run whose
     `session_id` differs from the path's `session_id` → 404). Use
     `record.landscape_run_id` as the join key into the Landscape DB.
     `get_run` raises `ValueError` if the run row does not exist —
     translate to 404 in the service.
   - (b) **Landscape read** via `RunLifecycleRepository.get_run(landscape_run_id)`
     (`src/elspeth/core/landscape/run_lifecycle_repository.py:262`), which
     is **synchronous** and takes a single `str` argument. It returns
     `Run | None`; `None` → `CorruptAuditRowError` (the composer
     `runs_table` row claimed a `landscape_run_id` that does not exist in
     the Landscape DB — that is a Tier-1 audit-database inconsistency).
   If the implementer finds the Landscape read surface needs extending
   (e.g., to expose a new column added by Task 7.0 that
   `RunLifecycleRepository.get_run` does not yet project), escalate to
   the operator — this is a meaningful scope addition.
2. The auth dep used by sibling `/api/sessions/{id}/...` routes —
   `get_current_user` from `auth/middleware.py`. The ownership-check
   helper is the shared `verify_session_ownership` free function in
   `sessions/ownership.py` (raises `HTTPException(404)` — same IDOR
   contract as Task 7.1).
3. The exact attribute names of the Landscape row corresponding to:
   `llm_call_count`, `source_data_hash` (the `output_file_hash` in our
   response model), `run_started_at`, `plugin_versions`, and the
   `seeded_from_cache` / `cache_key` metadata written by Task 7's
   `_replay_cached_content_to_landscape`. The audit-story service's
   field-presence check must match those exact names. These columns are
   added in Task 7.0 (see above) — verify they exist before implementing
   Step 4.

- [ ] **Step 2: Write the failing integration test.**

Create `tests/integration/web/test_audit_story_routes.py`:

```python
"""Integration tests for GET /api/sessions/{session_id}/runs/{run_id}/audit-story."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.pool import StaticPool

from elspeth.web.auth.middleware import get_current_user
from elspeth.web.auth.models import UserIdentity
from elspeth.web.sessions.engine import create_session_engine
from elspeth.web.sessions.routes import create_session_router
from elspeth.web.sessions.schema import initialize_session_schema


@pytest.fixture
def app(tmp_path: Path) -> FastAPI:
    engine = create_session_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    initialize_session_schema(engine)
    app = FastAPI()
    app.state.session_engine = engine
    # app.state.session_service: wired during recon (SessionServiceImpl).
    # app.state.run_lifecycle_repo: wired during recon (RunLifecycleRepository
    # against the Landscape DB). The service composes the two reads — there
    # is no `app.state.landscape` attribute.
    # app.state.settings: WebSettings(auth_provider="local", ...) — required
    # by verify_session_ownership.

    identity = UserIdentity(user_id="alice", username="alice")

    async def _mock_user() -> UserIdentity:
        return identity

    app.dependency_overrides[get_current_user] = _mock_user
    app.include_router(create_session_router())
    return app


def _stage_run_for_audit_story(
    app: FastAPI,
    *,
    session_id: str,
    run_id: str,
    user_id: str = "alice",
    llm_call_count: int = 5,
    output_file_hash: str = "cafebabe",
    run_started_at: datetime | None = None,
    plugin_versions: dict[str, str] | None = None,
    seeded_from_cache: bool = False,
    cache_key: str | None = None,
    omit_field: str | None = None,
) -> None:
    """Stage the THREE rows required to resolve an audit-story request.

    The service composes three reads in sequence (ownership → composer-DB
    run → Landscape-DB run). A test fixture must seed all three or the
    request will 404 partway through, masking the behaviour under test:

    1. ``sessions_table`` row (composer DB): keyed by ``session_id``,
       owned by ``user_id``, matching ``app.state.settings.auth_provider``.
       Required by ``verify_session_ownership``. Reuse the existing
       ``_make_session`` helper from
       ``tests/integration/web/conftest.py``.

    2. ``runs_table`` row (composer DB): keyed by ``run_id``, with
       ``session_id`` and ``landscape_run_id`` populated. The Tier-1
       invariant on ``RunRecord`` (``protocol.py:466``) requires
       ``landscape_run_id`` to be NOT NULL for terminal statuses
       (``completed`` / ``completed_with_failures`` / ``empty``); use
       ``status="completed"`` and assign ``landscape_run_id = "ldr-" + run_id``
       (or similar synthetic key) — fixtures don't need a real Landscape
       binding, just a consistent one.

    3. ``runs_table`` row (Landscape DB): keyed by the
       ``landscape_run_id`` from step 2, carrying the six audit-story
       columns added by Task 7.0. This is the row the
       ``RunLifecycleRepository.get_run`` lookup resolves.

    ``omit_field`` simulates Tier-1 corruption by writing the Landscape
    row WITHOUT the named column populated (or by setting the column to
    None when the schema permits — the service's field-presence check
    treats both as corruption for the required columns).

    Implementation depends on Step 1 recon for the exact insert patterns
    against both DBs.
    """
    raise NotImplementedError("filled in from Step 1 recon")


def test_get_audit_story_returns_real_landscape_data(app: FastAPI) -> None:
    """Response fields exactly match the Landscape row — no synthesis."""
    _stage_run_for_audit_story(
        app,
        session_id="sess-1",
        run_id="run-1",
        llm_call_count=5,
        output_file_hash="cafebabe1234",
        run_started_at=datetime(2026, 5, 15, 12, 0, tzinfo=UTC),
        plugin_versions={"web_scrape": "1.0.0", "llm_rate": "1.0.0"},
    )
    client = TestClient(app)
    response = client.get("/api/sessions/sess-1/runs/run-1/audit-story")
    assert response.status_code == 200
    body = response.json()
    assert body["run_id"] == "run-1"
    assert body["session_id"] == "sess-1"
    assert body["llm_call_count"] == 5
    assert body["output_file_hash"] == "cafebabe1234"
    assert body["run_started_at"] == "2026-05-15T12:00:00+00:00"
    assert body["plugin_versions"] == {"web_scrape": "1.0.0", "llm_rate": "1.0.0"}
    assert body["seeded_from_cache"] is False
    assert body["cache_key"] is None


def test_get_audit_story_for_cache_replay_surfaces_seeded_marker(
    app: FastAPI,
) -> None:
    """Cache-replay run → seeded_from_cache=true, cache_key is the SHA-256."""
    _stage_run_for_audit_story(
        app,
        session_id="sess-1",
        run_id="run-cache-replay",
        llm_call_count=0,  # cache replay: no live LLM calls
        seeded_from_cache=True,
        cache_key="a" * 64,
    )
    client = TestClient(app)
    response = client.get("/api/sessions/sess-1/runs/run-cache-replay/audit-story")
    assert response.status_code == 200
    body = response.json()
    assert body["seeded_from_cache"] is True
    assert body["cache_key"] == "a" * 64
    assert body["llm_call_count"] == 0


def test_get_audit_story_cross_session_returns_404(app: FastAPI) -> None:
    """run_id belongs to a different session → 404 (not 200, not 403).

    Same IDOR-safe contract as the cross-user case: an attacker who learns
    a foreign run_id (e.g. via an unrelated leak) cannot probe its
    existence by querying it under their own session_id. The service
    returns 404 whether the run is unknown or simply in a different
    session — the two cases are deliberately indistinguishable.
    """
    _stage_run_for_audit_story(app, session_id="sess-other", run_id="run-1")
    # Caller (alice) attempts to read sess-1's run-1, but run-1 lives in sess-other.
    _stage_run_for_audit_story(app, session_id="sess-1", run_id="run-2")  # alice's
    client = TestClient(app)
    response = client.get("/api/sessions/sess-1/runs/run-1/audit-story")
    assert response.status_code == 404


def test_get_audit_story_cross_user_returns_404(app: FastAPI) -> None:
    """Session not owned by current user → 404 (IDOR contract).

    Per the established IDOR contract (src/elspeth/web/sessions/ownership.py:33),
    cross-user access returns 404 (not 403) to avoid leaking session existence
    to an attacker enumerating UUIDs. Returning 403 would expose "this session
    exists, you just can't read it" vs "no such session". This is enforced by
    the shared `verify_session_ownership` helper, which raises
    HTTPException(404) on any access-control failure (unknown session, wrong
    user, wrong auth provider).
    """
    # Stage a session owned by bob, with a run.
    _stage_run_for_audit_story(
        app, session_id="sess-bob", run_id="run-b", user_id="bob"
    )
    client = TestClient(app)  # current user is alice (fixture)
    response = client.get("/api/sessions/sess-bob/runs/run-b/audit-story")
    assert response.status_code == 404


def test_get_audit_story_synthesis_forbidden(app: FastAPI) -> None:
    """Tier-1 invariant: missing required field → named exception → 500.

    The audit-story endpoint must NOT fabricate a default value to fill an
    absent field. Per CLAUDE.md "no inference - if it's not recorded, it
    didn't happen". A Landscape row missing llm_call_count is corruption,
    not an invitation to return 0.
    """
    _stage_run_for_audit_story(
        app,
        session_id="sess-1",
        run_id="run-broken",
        omit_field="llm_call_count",
    )
    client = TestClient(app)
    response = client.get("/api/sessions/sess-1/runs/run-broken/audit-story")
    assert response.status_code == 500
    # The error body names the missing field so an auditor can identify
    # the corrupt row without grepping logs.
    assert "llm_call_count" in response.json().get("detail", "")


def test_get_audit_story_unknown_run_returns_404(app: FastAPI) -> None:
    client = TestClient(app)
    response = client.get(
        "/api/sessions/sess-1/runs/nonexistent-run/audit-story"
    )
    assert response.status_code == 404
```

- [ ] **Step 3: Run test to verify it fails.**

```bash
.venv/bin/python -m pytest tests/integration/web/test_audit_story_routes.py -v
```

Expected: FAIL.

- [ ] **Step 4: Implement Pydantic model, service, and route.**

Append to `src/elspeth/web/sessions/schemas.py`:

```python
class RunAuditStoryResponse(BaseModel):
    """Audit-story response for GET /api/sessions/{id}/runs/{run_id}/audit-story.

    All fields are read from a real Landscape audit row. No field is ever
    synthesised or defaulted. Absence of any field below in the underlying
    row is Tier-1 corruption (CorruptAuditRowError → 500).
    """

    model_config = ConfigDict(strict=True, extra="forbid")

    run_id: str
    session_id: str
    llm_call_count: int
    output_file_hash: str  # the row's source_data_hash
    run_started_at: datetime
    plugin_versions: dict[str, str]
    seeded_from_cache: bool
    cache_key: str | None
```

Create `src/elspeth/web/sessions/audit_story_service.py`:

```python
"""Audit-story service — read-only Landscape projection for a single run.

Phase 4A.7.2. The frontend's getRunAuditSummary (21b2 Task 9) calls the
route that wraps this service.

Tier model:
- Inputs (session_id, run_id, user): Tier 3 trust — validated via the
  shared `verify_session_ownership` helper before touching the Landscape.
  The helper raises HTTPException(404) on any access-control failure
  (IDOR contract; sessions/ownership.py:33).
- Landscape rows: Tier 1 — every required field MUST be present. Absence
  → CorruptAuditRowError → 500.

The Landscape data is fetched via a TWO-read composition:
  1. composer DB: ``session_service.get_run(UUID(run_id))`` resolves the
     ``(session_id, landscape_run_id)`` binding.
  2. Landscape DB: ``run_lifecycle_repo.get_run(landscape_run_id)`` (sync,
     single arg) returns the row with the audit-story columns.

Per CLAUDE.md no-defensive-programming: NO synthesis, NO `.get(field, default)`,
NO `getattr(row, field, None)`. Direct attribute access only. Absent fields
raise the named exception with `from exc` chaining so an auditor sees which
field was missing.
"""

from __future__ import annotations

from uuid import UUID

from fastapi import HTTPException, Request

from elspeth.core.landscape.run_lifecycle_repository import RunLifecycleRepository
from elspeth.web.auth.models import UserIdentity
from elspeth.web.sessions.ownership import verify_session_ownership
from elspeth.web.sessions.protocol import SessionServiceProtocol
from elspeth.web.sessions.schemas import RunAuditStoryResponse


class CorruptAuditRowError(Exception):
    """Raised when the Landscape row is missing a required field."""

    def __init__(self, *, run_id: str, missing_field: str) -> None:
        super().__init__(
            f"audit row for run {run_id!r} is missing required field "
            f"{missing_field!r}; this is Tier-1 corruption"
        )
        self.run_id = run_id
        self.missing_field = missing_field


async def get_run_audit_story(
    *,
    session_id: str,
    run_id: str,
    user: UserIdentity,
    request: Request,
    session_service: SessionServiceProtocol,
    run_lifecycle_repo: RunLifecycleRepository,
) -> RunAuditStoryResponse:
    """Return the audit-story projection for ``(session_id, run_id)``.

    Authorization order (see Authorization order section above):
      1. Caller authenticated — enforced upstream by the route dep.
      2. Caller owns ``session_id`` — ``verify_session_ownership`` raises
         ``HTTPException(404)`` on mismatch (IDOR contract;
         ``sessions/ownership.py:33``).
      3. ``run_id`` belongs to ``session_id`` — composer-DB read.
      4. Landscape row exists and is complete — Landscape read + Tier-1
         field-presence check.

    The Landscape data is composed from TWO reads (no
    ``app.state.landscape`` shortcut exists):
      (a) composer DB: ``session_service.get_run(UUID(run_id))`` →
          ``RunRecord`` with ``session_id`` (for the cross-session check)
          and ``landscape_run_id`` (the join key);
      (b) Landscape DB: ``run_lifecycle_repo.get_run(landscape_run_id)``
          → the ``Run`` row carrying the audit-story columns added by
          Task 7.0.

    Raises:
        HTTPException(404): session not owned by caller (from
            ``verify_session_ownership``), run not found in the composer
            DB, run belongs to a different session, or the Landscape row
            for the run is missing.
        CorruptAuditRowError: the Landscape row is missing a Tier-1
            required field (→ 500).
    """
    # 1. Ownership check. Raises HTTPException(404) on mismatch — IDOR-safe.
    await verify_session_ownership(
        session_id=UUID(session_id),
        user=user,
        request=request,
    )

    # 2. Composer-DB read: locate the run, verify it belongs to this session.
    try:
        record = await session_service.get_run(UUID(run_id))
    except ValueError:
        # Composer-DB has no row for this run_id.
        raise HTTPException(
            status_code=404, detail=f"Run {run_id!r} not found"
        ) from None
    if str(record.session_id) != session_id:
        # Run exists but in a different session. Return 404 (IDOR-safe):
        # do not reveal that the run exists elsewhere.
        raise HTTPException(
            status_code=404,
            detail=f"Run {run_id!r} not found in session {session_id!r}",
        )
    if record.landscape_run_id is None:
        # The composer DB has no Landscape join key yet — the run did not
        # reach the engine-completion path that writes landscape_run_id.
        # That is not a Tier-1 corruption, it just means there is no
        # audit story to read.
        raise HTTPException(
            status_code=404,
            detail=f"Run {run_id!r} has no Landscape audit row yet",
        )

    # 3. Landscape read. Synchronous single-arg API on the repository.
    landscape_row = run_lifecycle_repo.get_run(record.landscape_run_id)
    if landscape_row is None:
        # The composer-DB row claimed a landscape_run_id that does not
        # exist in the Landscape DB. That is a Tier-1 cross-database
        # inconsistency: the audit trail is broken.
        raise CorruptAuditRowError(
            run_id=run_id, missing_field="<landscape row missing entirely>"
        )

    # 4. Tier-1 field-presence check. Direct attribute access — no
    # `getattr(row, field, default)`, no `.get(...)`, no synthesis.
    for required in (
        "llm_call_count",
        "source_data_hash",
        "run_started_at",
        "plugin_versions",
        "seeded_from_cache",
        "cache_key",
    ):
        if not _row_has_field(landscape_row, required):
            raise CorruptAuditRowError(run_id=run_id, missing_field=required)

    return RunAuditStoryResponse(
        run_id=run_id,
        session_id=session_id,
        llm_call_count=landscape_row.llm_call_count,
        output_file_hash=landscape_row.source_data_hash,
        run_started_at=landscape_row.run_started_at,
        plugin_versions=landscape_row.plugin_versions,
        seeded_from_cache=landscape_row.seeded_from_cache,
        cache_key=landscape_row.cache_key,
    )


def _row_has_field(row: object, field: str) -> bool:
    """Check whether ``row`` carries the named attribute.

    Detects column-missing (corruption), NOT value-is-None (which is legal
    for ``cache_key`` on a live run). Implementation depends on the exact
    shape returned by ``RunLifecycleRepository.get_run`` (likely a frozen
    dataclass — confirm during Step 1 recon).

    NOTE: do NOT use ``hasattr()`` here — per CLAUDE.md it is
    unconditionally banned (it swallows arbitrary ``@property`` exceptions).
    Use ``field in vars(row)`` for dataclass rows or
    ``field in row._fields`` for NamedTuple rows; pick the form that
    matches the actual return type.
    """
    raise NotImplementedError("filled in from Step 1 recon")
```

Add the route handler to `src/elspeth/web/sessions/routes.py` alongside the
existing `update_session` PATCH route:

```python
@router.get(
    "/{session_id}/runs/{run_id}/audit-story",
    response_model=RunAuditStoryResponse,
)
async def get_run_audit_story_route(
    session_id: str,
    run_id: str,
    request: Request,
    user: UserIdentity = Depends(get_current_user),
) -> RunAuditStoryResponse:
    from elspeth.web.sessions.audit_story_service import (
        CorruptAuditRowError,
        get_run_audit_story,
    )

    # The service reads `session_service` and the Landscape repository
    # from app.state. There is no `app.state.landscape` attribute — the
    # Landscape read uses `RunLifecycleRepository` registered on app.state
    # under its own attribute (confirm exact attribute name during recon —
    # the project convention is `app.state.run_lifecycle_repo`).
    # Ownership failures (HTTPException(404) from verify_session_ownership)
    # and "run not found"/"run-in-other-session" cases (HTTPException(404)
    # raised inside the service) propagate directly to FastAPI's handler.
    try:
        return await get_run_audit_story(
            session_id=session_id,
            run_id=run_id,
            user=user,
            request=request,
            session_service=request.app.state.session_service,
            run_lifecycle_repo=request.app.state.run_lifecycle_repo,
        )
    except CorruptAuditRowError as exc:
        # Tier-1 corruption: surface the field name in the 500 body so an
        # auditor reading the response knows exactly which column is broken.
        raise HTTPException(status_code=500, detail=str(exc)) from exc
```

- [ ] **Step 5: Run test to verify it passes.**

```bash
.venv/bin/python -m pytest tests/integration/web/test_audit_story_routes.py -v
```

Expected: PASS — all six tests green, including the no-synthesis test.

- [ ] **Step 6: Run the full integration suite.**

```bash
.venv/bin/python -m pytest tests/integration/web/ -v
```

Expected: PASS.

- [ ] **Step 7: Commit.**

```bash
git add src/elspeth/web/sessions/audit_story_service.py \
        src/elspeth/web/sessions/routes.py \
        src/elspeth/web/sessions/schemas.py \
        tests/integration/web/test_audit_story_routes.py
git commit -m "feat(web): GET /api/sessions/{id}/runs/{run_id}/audit-story (Phase 4A.7.2)"
```

---

## Task 7.3: Frontend client functions — `runTutorialPipeline`, `getRunAuditSummary`, `renameSession`

**Files:**

- Modify: `src/elspeth/web/frontend/src/api/client.ts` — three function additions / one rename.
- Modify: `src/elspeth/web/frontend/src/api/client.test.ts` (or create per existing per-feature test convention — see `client.preferences.test.ts`, `client.recovery.test.ts`).
- Create: `src/elspeth/web/frontend/src/api/client.tutorial.test.ts` — tests for the three new functions (matches the per-feature test-file convention live in the repo).

The frontend tasks in 21b2 (Tasks 8, 9, 10) spy on three `client.ts`
symbols that do not yet exist. Task 7.3 lands those symbols against the
backend endpoints introduced by Tasks 7.1 and 7.2 plus the **already-extant**
session-update endpoint.

**Pre-existing surface (confirmed by 21a recon, 2026-05-19):**

- The backend already exposes `PATCH /api/sessions/{id}` with body
  `{title: str}`. The current `client.ts` exports `updateSessionTitle`
  (line 328 at recon time) which calls this endpoint. 21b2 Task 10 uses
  the name `renameSession` for the same wire endpoint. **No backend
  route change is needed.** Per CLAUDE.md no-legacy-code: rename
  `updateSessionTitle` → `renameSession` (single rename, all call sites
  updated in the same commit; no shim).

- `runTutorialPipeline` and `getRunAuditSummary` are new functions
  consuming the newly-defined backend routes (Tasks 7.1 and 7.2).

- [ ] **Step 1: Reconnaissance — confirm current client.ts surface and call sites.**

```bash
grep -n "updateSessionTitle\|runTutorialPipeline\|getRunAuditSummary\|renameSession" \
  src/elspeth/web/frontend/src/ -r
```

Catalogue every call site of `updateSessionTitle` (these are the
rename-target files Task 7.3 will edit). Confirm that no other consumer
relies on the name `updateSessionTitle`.

- [ ] **Step 2: Write the failing test.**

Create `src/elspeth/web/frontend/src/api/client.tutorial.test.ts`:

```typescript
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import {
  runTutorialPipeline,
  getRunAuditSummary,
  renameSession,
} from './client';

describe('client.tutorial — runTutorialPipeline', () => {
  beforeEach(() => {
    global.fetch = vi.fn();
  });
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('POSTs to /api/tutorial/run with session_id + prompt', async () => {
    (global.fetch as ReturnType<typeof vi.fn>).mockResolvedValue(
      new Response(
        JSON.stringify({
          run_id: 'r1',
          output: {
            rows: [{ url: 'a', score: 5 }],
            source_data_hash: 'a7f3e2',
          },
          seeded_from_cache: false,
          cache_key: null,
        }),
        { status: 200 },
      ),
    );
    const result = await runTutorialPipeline({
      session_id: 'sess-1',
      prompt: 'rate these',
    });
    expect(global.fetch).toHaveBeenCalledWith(
      '/api/tutorial/run',
      expect.objectContaining({
        method: 'POST',
        body: JSON.stringify({ session_id: 'sess-1', prompt: 'rate these' }),
      }),
    );
    expect(result.run_id).toBe('r1');
    expect(result.seeded_from_cache).toBe(false);
  });

  it('surfaces backend error status as a thrown error', async () => {
    (global.fetch as ReturnType<typeof vi.fn>).mockResolvedValue(
      new Response('{"detail":"unknown session"}', { status: 404 }),
    );
    await expect(
      runTutorialPipeline({ session_id: 'bad', prompt: 'x' }),
    ).rejects.toThrow();
  });
});

describe('client.tutorial — getRunAuditSummary', () => {
  beforeEach(() => {
    global.fetch = vi.fn();
  });
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('GETs /api/sessions/{id}/runs/{run_id}/audit-story', async () => {
    (global.fetch as ReturnType<typeof vi.fn>).mockResolvedValue(
      new Response(
        JSON.stringify({
          run_id: 'r1',
          session_id: 'sess-1',
          llm_call_count: 5,
          output_file_hash: 'cafe',
          run_started_at: '2026-05-15T12:00:00Z',
          plugin_versions: { web_scrape: '1.0.0' },
          seeded_from_cache: false,
          cache_key: null,
        }),
        { status: 200 },
      ),
    );
    const summary = await getRunAuditSummary('sess-1', 'r1');
    expect(global.fetch).toHaveBeenCalledWith(
      '/api/sessions/sess-1/runs/r1/audit-story',
      expect.objectContaining({ headers: expect.anything() }),
    );
    expect(summary.llm_call_count).toBe(5);
  });

  it('surfaces 500 (corrupt audit row) as a thrown error', async () => {
    (global.fetch as ReturnType<typeof vi.fn>).mockResolvedValue(
      new Response('{"detail":"missing llm_call_count"}', { status: 500 }),
    );
    await expect(getRunAuditSummary('s', 'r')).rejects.toThrow();
  });
});

describe('client.tutorial — renameSession', () => {
  beforeEach(() => {
    global.fetch = vi.fn();
  });
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('PATCHes /api/sessions/{id} with {title}', async () => {
    (global.fetch as ReturnType<typeof vi.fn>).mockResolvedValue(
      new Response(
        JSON.stringify({ id: 'sess-1', title: 'hello-world (cool government pages)' }),
        { status: 200 },
      ),
    );
    const result = await renameSession(
      'sess-1',
      'hello-world (cool government pages)',
    );
    expect(global.fetch).toHaveBeenCalledWith(
      '/api/sessions/sess-1',
      expect.objectContaining({
        method: 'PATCH',
        body: JSON.stringify({ title: 'hello-world (cool government pages)' }),
      }),
    );
    expect(result.title).toBe('hello-world (cool government pages)');
  });
});
```

- [ ] **Step 3: Run test to verify it fails.**

```bash
cd src/elspeth/web/frontend && npm test -- client.tutorial.test.ts
```

Expected: FAIL — `runTutorialPipeline`, `getRunAuditSummary`, `renameSession`
not exported.

- [ ] **Step 4: Implement.**

In `src/elspeth/web/frontend/src/api/client.ts`:

1. **Add new types** alongside the existing type declarations (or in a new
   `tutorialTypes.ts` if `client.ts` already exceeds the project's
   conventional module-size threshold — recon decides):

   ```typescript
   export interface TutorialRunRequest {
     session_id: string;
     prompt: string;
   }

   export interface TutorialRunOutput {
     rows: Array<Record<string, unknown>>;
     source_data_hash: string;
   }

   export interface TutorialRunResponse {
     run_id: string;
     output: TutorialRunOutput;
     seeded_from_cache: boolean;
     cache_key: string | null;
   }

   export interface RunAuditStoryResponse {
     run_id: string;
     session_id: string;
     llm_call_count: number;
     output_file_hash: string;
     run_started_at: string;
     plugin_versions: Record<string, string>;
     seeded_from_cache: boolean;
     cache_key: string | null;
   }
   ```

2. **Add `runTutorialPipeline`**:

   ```typescript
   /** Run the tutorial pipeline. Backend may serve a cached replay; the
    * returned run_id is ALWAYS owned by the current session — see 21a
    * §"New endpoints" for the cache-replay contract.
    */
   export async function runTutorialPipeline(
     body: TutorialRunRequest,
   ): Promise<TutorialRunResponse> {
     const response = await fetch('/api/tutorial/run', {
       method: 'POST',
       headers: authHeaders('application/json'),
       body: JSON.stringify(body),
     });
     return parseResponse<TutorialRunResponse>(response);
   }
   ```

3. **Add `getRunAuditSummary`**:

   ```typescript
   /** Read the audit-story for a (session_id, run_id) pair.
    *
    * All fields are real audit data — no synthesis. A 500 response means
    * the audit row is Tier-1 corrupt; surface the error rather than
    * fabricating defaults (per CLAUDE.md no-defensive-programming).
    */
   export async function getRunAuditSummary(
     sessionId: string,
     runId: string,
   ): Promise<RunAuditStoryResponse> {
     const response = await fetch(
       `/api/sessions/${sessionId}/runs/${runId}/audit-story`,
       { headers: authHeaders() },
     );
     return parseResponse<RunAuditStoryResponse>(response);
   }
   ```

4. **Rename `updateSessionTitle` → `renameSession`.** Per CLAUDE.md "No
   Legacy Code Policy", do not leave an alias. The existing function body
   (PATCH `/api/sessions/${sessionId}` with `{title}`) is structurally
   correct — just rename. Update every call site discovered in Step 1's
   `grep`. Atomic single commit.

   ```typescript
   /** Update the user-visible title for a session.
    *
    * Used by the tutorial finalisation flow (21b2 Task 10) and by any
    * other UI that lets a user rename a session. Wire endpoint:
    * PATCH /api/sessions/{id} with body {title}.
    */
   export async function renameSession(
     sessionId: string,
     title: string,
   ): Promise<Session> {
     const response = await fetch(`/api/sessions/${sessionId}`, {
       method: 'PATCH',
       headers: authHeaders('application/json'),
       body: JSON.stringify({ title }),
     });
     return parseResponse<Session>(response);
   }
   ```

- [ ] **Step 5: Update all call sites of the renamed function.**

```bash
grep -rn "updateSessionTitle" src/elspeth/web/frontend/src/ --include="*.ts" --include="*.tsx"
```

Replace every hit with `renameSession`. Run TypeScript compile to confirm
no stragglers:

```bash
cd src/elspeth/web/frontend && npx tsc --noEmit
```

- [ ] **Step 6: Run test to verify it passes.**

```bash
cd src/elspeth/web/frontend && npm test -- client.tutorial.test.ts
```

Expected: PASS.

- [ ] **Step 7: Run the full frontend test suite to catch regressions.**

```bash
cd src/elspeth/web/frontend && npm test
```

Expected: PASS — the rename's call-site updates do not break existing tests.

- [ ] **Step 8: Commit.**

```bash
git add src/elspeth/web/frontend/src/api/client.ts \
        src/elspeth/web/frontend/src/api/client.tutorial.test.ts \
        <renamed-call-sites>
git commit -m "feat(frontend): runTutorialPipeline + getRunAuditSummary + rename updateSessionTitle (Phase 4A.7.3)"
```

---

## What Phase 4A leaves the backend in

After Tasks 0–7.3: new column, Tier-1 guards, cache module (filesystem-backed,
corruption-detecting), run-path cache consult, **tutorial-run route** (POST
/api/tutorial/run, Task 7.1) with completed-user / freeform-user bypass,
**audit-story route** (GET …/audit-story, Task 7.2) with no-synthesis Tier-1
invariant, and **frontend client functions** (`runTutorialPipeline`,
`getRunAuditSummary`, renamed `renameSession`, Task 7.3). Phase 4B wires
the frontend components against these surfaces.

## Risks and mitigations

Key risks: run-path entry point not where assumed (Task 7 Step 1 recon resolves); cache fires for non-tutorial users (gate on `tutorial_completed_at is None` + regression test); model_id derivation drifts (integration tests assert hit-on-pre-populated-cache); concurrent writers (atomic via `os.replace`); audit-story synthesis (no-synthesis Tier-1 test `test_get_audit_story_synthesis_forbidden` in Task 7.2 pins the invariant); `_is_canonical_seed_pipeline` force-live flag collides with Task 7's existing canonical-seed shape check (Task 7.1 Step 5 adds the flag check at the top of the helper; Task 7's `test_non_tutorial_user_skips_cache` and `test_edited_prompt_skips_cache` must remain green after Task 7.1 lands — verified by Step 7); renamed `updateSessionTitle` call sites missed (Task 7.3 Step 5 runs `tsc --noEmit` to catch).

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
