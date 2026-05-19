# Phase 4A — Backend Part 1: infrastructure (Tasks 0–7.0, 7)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Status:** 2026-05-19. Part 1 of the Phase 4 backend plan. The original `21a-phase-4-backend.md` was split once it exceeded 4,900 lines; **endpoint surface + telemetry live in [21a2-phase-4-backend-part-2.md](21a2-phase-4-backend-part-2.md)**.

**Goal:** Land the backend infrastructure half of Phase 4 — extend `user_preferences_table`
with `tutorial_completed_at`, extend `PreferencesService` and the
`/api/composer-preferences` route to expose it, add a flat-file tutorial cache
keyed by `(canonical_prompt_sha256, model_id)`, extend `runs_table` with the
audit-story columns, ship the `LandscapeWriteRepository`, and wire the cache
into the composer run path so that tutorial-mode runs of the canonical seed
return cached output deterministically. Part 2 (21a2) layers the
tutorial-run endpoint, the audit-story endpoint, the frontend API client,
and the launch telemetry counters on top of this infrastructure.

**Architecture:** Schema-then-Pydantic-then-service-then-route-then-cache-
then-integration. Tests are TDD-shaped at every step. The new code paths reuse
all existing infrastructure (the Phase 1A `PreferencesService`, the existing
run-path, the existing Landscape readers); Phase 4A only adds the column, the
cache module, and a single call-site that consults the cache before invoking
the LLM.

**Tech Stack:** SQLAlchemy Core, FastAPI, Pydantic v2, pytest, hashlib (stdlib).

**Sibling plans:**

- [21a2-phase-4-backend-part-2.md](21a2-phase-4-backend-part-2.md) — endpoint surface (`POST /api/tutorial/run`, `GET …/audit-story`, `DELETE /api/tutorial/orphans` per Systems R2-S5) and launch telemetry counters (Task 8). The frontend API client functions originally drafted as 21a2 Task 7.3 are relocated to 21b2 §"Task 7.5" — those symbols live in `client.ts` and belong with the frontend plan.
- [21b1-phase-4-frontend-part-1.md](21b1-phase-4-frontend-part-1.md) — frontend store, copy module, Turns 1–3.
- [21b2-phase-4-frontend-part-2.md](21b2-phase-4-frontend-part-2.md) — frontend Turns 4–6, container, detection, integration, smoke.

**Overview document:** [21-phase-4-hello-world-tutorial.md](21-phase-4-hello-world-tutorial.md).

**PR mapping:** This plan ships as part of **PR-21a** alongside 21a2 (21a1 + 21a2 are co-dependent and land in a single PR). PR-21a must
merge to `RC5.2` **before** PR-21b (the frontend half from 21b1 + 21b2) —
see overview §"PR strategy" for rationale.

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
  - Filesystem-backed (configurable directory via the validated
    `WebSettings.tutorial_cache_dir` field; defaults to
    `<data_dir>/tutorial_cache/` resolved against the validated
    `WebSettings.data_dir`).
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

Consumed by Phase 4B Part 2. Contracts are declared here so 21b2 can reference them; **implementation tasks live in [21a2-phase-4-backend-part-2.md](21a2-phase-4-backend-part-2.md)**: **Task 7.1** (POST /api/tutorial/run), **Task 7.2** (GET …/audit-story), **Task 7.3** (frontend `client.ts` functions consuming both).

**`POST /api/tutorial/run`** — body: `{"session_id": "<uuid>", "prompt": "<canonical-or-edited-seed>"}`. Response: `{"run_id": "<uuid>", "output": {"rows": [...], "source_data_hash": "<hex>"}, "seeded_from_cache": <bool>, "cache_key": "<hex>" | null}`. Cache-hit: the backend **synthesises a real Landscape entry under the current user's session** via `_replay_cached_content_to_landscape` (defined in Task 7 of this Part 1), populated from the cached content (rows, source_data_hash, llm_call_count=0, pipeline_yaml) plus a `seeded_from_cache: true` marker carrying the cache key. The returned `run_id` is **owned by the current session** — there is no foreign-run reference in the response. Cache-miss: live run (~30s), populates cache on success. Cache is bypassed (live run, no consult, no write) when the user's `tutorial_completed_at IS NOT NULL` (post-completion) **or** when their `default_mode == 'freeform'` (freeform users skip tutorial caching entirely). Unknown session → 404. Tier-1 corruption on the caller's preferences row → 500 (`CorruptPreferencesError` propagates to the global handler). Defined by Task 7.1 (21a2).

**`GET /api/sessions/{session_id}/runs/{run_id}/audit-story`** — response: `{"run_id": "<uuid>", "session_id": "<uuid>", "llm_call_count": N, "output_file_hash": "<hex>", "started_at": "<iso8601>", "plugin_versions": {...}, "seeded_from_cache": <bool>, "cache_key": "<hex>" | null}`. Reads **entirely from real Landscape audit rows** — **no field is ever synthesised or defaulted**. When the run was a cache hit, `seeded_from_cache` is `true`, `llm_call_count` is `0`, and `cache_key` is the SHA-256 that points at the original cache-seeding run for cross-run lineage joins. Run not in session → 404. Session not owned by caller → 404 (IDOR contract: never 403, to avoid leaking session existence — see `src/elspeth/web/sessions/ownership.py:33`). Tier-1 corruption (audit row missing a required field such as `llm_call_count`) → 500 (named exception). Landscape failure propagates — no fallback (design doc 04: "Otherwise the demonstration is theatre."). Defined by Task 7.2 (21a2).

## Trust tier check (per CLAUDE.md)

| Surface | Tier | Handling |
|---|---|---|
| Inbound `tutorial_completed_at` (PATCH body) | Tier 3 | Pydantic rejects non-datetime with 422. |
| Outbound `tutorial_completed_at` (DB read) | Tier 1 | `_row_to_prefs` guards: must be `None` or `datetime`; non-datetime → crash. |
| Tutorial cache file contents | server-generated content cache | Parse failure = corruption → crash. Operationally follows Tier-1 rules (crash on corruption with file path + parse error chained via `from`; miss on absence; no live-LLM fallback on corruption). Conceptually the data is LLM-derived, not Tier-1 "our data" — the crash-on-corrupt invariant exists because we wrote the file and corruption indicates a fault we must surface, not because we own the source-of-truth data. See Task 5 §"Operational guarantees" for the full framing. |
| Tutorial cache file presence | n/a | Absent = miss, not fault. |
| Canonical seed prompt | constant | Python constant shared with frontend; drift → cache miss (intended). |
| LLM results in cache | server-generated cache content | Cache write happens after the canonical-seed run is recorded in the Landscape and only when every row succeeded (P18 `_all_rows_succeeded(result)` gate; see Task 7's run-path code block). Corruption → crash on parse. |

## File structure

The Phase 4A backend touches both Part-1 (infrastructure) and Part-2 (endpoint + telemetry) files. Part-1 paths are listed below; Part-2 paths (tutorial-run routes/service, audit-story routes/service, frontend client, telemetry emit sites) live in 21a2's §"File structure".

**New (Part 1 — infrastructure):**

- `src/elspeth/web/preferences/tutorial_cache.py` — cache module.
- `tests/unit/web/preferences/test_tutorial_cache.py` — cache unit tests.
- `tests/integration/web/test_tutorial_cache_run_integration.py` — cache wiring.

**Modified (Part 1 — infrastructure):**

- `src/elspeth/web/sessions/models.py` — add `tutorial_completed_at` column.
- `src/elspeth/web/preferences/models.py` — extend Pydantic models.
- `src/elspeth/web/preferences/service.py` — extend read/write code paths.
- `tests/unit/web/preferences/test_schema.py` — extend expected-columns set.
- `tests/unit/web/preferences/test_models.py` — extend Pydantic tests.
- `tests/unit/web/preferences/test_service.py` — extend service tests.
- `tests/integration/web/test_preferences_routes.py` — extend route tests.
- The composer run-path file (identified during Task 7) — wire cache consult; Part-2 Task 7.1 extends `_is_canonical_seed_pipeline` with a force-live escape.

**Not modified (Part 1):**

- `src/elspeth/web/preferences/routes.py` — Pydantic-model extension propagates
  automatically through `response_model`.

**Part 2 surface (see 21a2 §"File structure" for the full enumeration):** `src/elspeth/web/composer/tutorial_run_routes.py`, `src/elspeth/web/composer/tutorial_service.py`, `src/elspeth/web/sessions/audit_story_service.py`, `src/elspeth/web/sessions/routes.py` (`GET …/audit-story` handler), `src/elspeth/web/sessions/schemas.py` (`RunAuditStoryResponse`), `src/elspeth/web/frontend/src/api/client.ts` (`runTutorialPipeline`, `getRunAuditSummary`, `renameSession` rename), and the FastAPI app-composition site (`include_router(create_tutorial_run_router())`).

## Database migration note (operator action)

Task 1 and Task 7.0 each require a DB-delete before new code serves
traffic — Task 1 on the **sessions DB** (because `SESSION_SCHEMA_EPOCH`
bumps and `user_preferences_table` gains a column), Task 7.0 on the
**Landscape audit DB** (because `runs_table` gains three columns:
`llm_call_count`, `seeded_from_cache`, `cache_key` — R2-S4 final list,
2026-05-19).
Phase 4B's smoke task performs both. If Phase 4A ships independently,
operator must perform both deletes first; they are independent
operations on independent files. All users' `tutorial_completed_at`
resets to NULL — every user retakes the tutorial on next login. See
§"DB-delete cadence" for the full sequence context, including the
table enumerating both events.

### Cache warming (post-deploy, post-restart)

After the sessions DB delete and `systemctl restart elspeth-web.service`,
the operator MUST warm the tutorial cache before the first production
user hits the tutorial. Without warming, that first user pays the full
~30-second LLM cost on every fresh deploy (the cache directory is empty
until something populates it).

Run from the deployment host, against the deployed model configuration:

```bash
elspeth tutorial warm-cache
```

This fires the canonical seed prompt (the same constant used by Task 5
and the frontend copy module) through the same run-path that an
interactive tutorial user would exercise, and writes the resulting
cache entry into `<data_dir>/tutorial_cache/` via the same
`TutorialCache.store(...)` code path (Task 5 + Task 7). On success the
next live user hits the cache and pays nothing.

**Cache entries are environment-specific.** The cache key is
`SHA-256(canonical_prompt + ":" + composer_model + ":" + transform_model)`
— see Task 7 Step 2's `_model_id_for_pipeline` (P19). Staging-cache
**cannot be promoted to prod** by copying files between hosts: the two
environments have different model configurations, so a copied entry's
key would not match a prod-side `lookup(...)` and the file would
either be ignored (miss) or trigger the Task 5 §"Operational guarantees"
mismatch crash (file in the wrong location for its recorded
`(canonical_prompt, model_id)`). The warm-cache step must run in
**each environment independently**, after each DB-delete + restart.

If the canonical pipeline's model configuration changes (a new
composer model, a new transform model, or both), the warm-cache step
re-runs against the new configuration; the old entry stops matching
the new key and is dead weight until an operator clears the directory.

The CLI command is a thin wrapper over the existing run-path; the
implementation work is the wrapper itself, owned by Task 5 (which
already specifies the cache store path the wrapper drives). The
deployment runbook entry is owned by this section. The warm step is
**mandatory** on every fresh deploy that included a sessions DB
delete; the smoke task in 21b2 verifies it executed.

## Verification approach

Each task is TDD-shaped (failing test, run-to-fail, implement, run-to-pass,
commit). Part 1 (this document) lands Tasks 0–7.0 and Task 7; Part 2 (21a2)
lands Tasks 7.1, 7.2, 7.3, and 8. After all 21a tasks plus the Phase 4B
work land, the Phase 4B integration tests and Playwright smoke exercise the
routes and the cache wiring end-to-end. The Phase 4B smoke task performs
the operator DB-deletes and re-runs the full test suite.

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
#        PATCH semantics" in `21a1-phase-4-backend-part-1.md`). Operators
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
this commit. See `21a1-phase-4-backend-part-1.md` §"DB-delete cadence:
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
- Modify: `src/elspeth/web/config.py` — extend `WebSettings` with the
  `tutorial_cache_dir` field (load-bearing prerequisite for Task 5).
- Modify: `tests/unit/web/preferences/test_models.py`.

> **Required prerequisite (Reality finding R2-6, 2026-05-19).**
> Task 2 also extends `WebSettings` (in `src/elspeth/web/config.py`) with:
>
> ```python
> # Phase 4A: cache directory for the tutorial-seed run cache. Defaults to
> # ``<data_dir>/tutorial_cache/`` resolved against the validated
> # ``WebSettings.data_dir`` (see model_validator below). Operators can
> # override via ``ELSPETH_WEB__TUTORIAL_CACHE_DIR``.
> tutorial_cache_dir: Path | None = Field(default=None)
> ```
>
> AND a `model_validator(mode='after')` that defaults the field to
> `data_dir / "tutorial_cache"` when unset. Without this field, Task 5's
> cache tests fail at WebSettings instantiation with
> `"Extra inputs are not permitted"` (the live `WebSettings` uses
> `ConfigDict(extra='forbid')`), and Task 6's app-composition site has no
> validated path to pass into `TutorialCache(cache_dir=...)`. This is the
> P8 resolution (no hardcoded absolute default path; flow through the
> validated `WebSettings`).

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


def test_update_composer_preferences_request_extra_field_rejected() -> None:
    """Pydantic config: extra='forbid' must reject unexpected fields.

    Pins the strict contract — a malformed PATCH body with extra fields
    should be 422-rejected, not silently coerced or partially applied.
    Phase 4 adds `tutorial_completed_at` to the model; this test guards
    against a future refactor that quietly drops `extra='forbid'` and
    starts accepting typos like `tutorial_complete` or `tutorialCompletedAt`.
    """
    with pytest.raises(ValidationError) as exc_info:
        UpdateComposerPreferencesRequest(
            default_mode="guided",
            rogue_field="should be rejected",  # type: ignore[call-arg]
        )
    assert "rogue_field" in str(exc_info.value)
```

- [ ] **Step 2: Run to fail.** `.venv/bin/python -m pytest tests/unit/web/preferences/test_models.py -v` → FAIL (`tutorial_completed_at` not a field on the Pydantic models).

- [ ] **Step 3: Extend the Pydantic models.**

In `src/elspeth/web/preferences/models.py`:

```python
class ComposerPreferences(BaseModel):
    """The full preferences payload returned by GET and PATCH.

    ``updated_at`` is nullable (Panel U1): when no DB row exists for
    the user, the response represents the in-server *default* — there
    is no write event to attach a timestamp to, and fabricating
    ``self._now()`` would put a value the system never wrote into an
    audit-visible field (CLAUDE.md fabrication test). The no-row GET
    path and the empty-PATCH-on-no-row path both return
    ``updated_at=None``; every other response returns the real write
    time.
    """

    model_config = ConfigDict(strict=True, extra="forbid")

    default_mode: ComposerMode
    banner_dismissed_at: datetime | None
    # Phase 4: NULL = user is in tutorial mode. Non-NULL = tutorial complete.
    tutorial_completed_at: datetime | None = None
    updated_at: datetime | None


class UpdateComposerPreferencesRequest(BaseModel):
    """Partial-update payload for PATCH.

    Uses ``ConfigDict(extra='forbid')`` only — not ``strict=True`` —
    because ``strict=True`` on a request body with ``datetime`` fields
    would reject the standard JSON ISO-8601 string representation
    (Pydantic v2 strict mode disallows string→datetime coercion). The
    Tier-3 boundary contract is "validate, coerce where the wire format
    permits, never fabricate"; rejecting the JSON datetime wire shape
    is too aggressive for the request side.
    """

    model_config = ConfigDict(extra="forbid")

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

The response model uses ``strict=True, extra='forbid'``; the request
model uses ``extra='forbid'`` only. This asymmetry is deliberate and
matches the live shape in `src/elspeth/web/preferences/models.py` — see
the docstring update referenced in Task 3's implementer note.

- [ ] **Step 4: Run test to verify it passes.**

```bash
.venv/bin/python -m pytest tests/unit/web/preferences/test_models.py -v
```

Expected: PASS — all model tests green (existing + 5 new).

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
> **Fixture name and location (Reality finding R2-13, 2026-05-19).** Match
> the canonical name used by the telemetry conftest
> (`tests/unit/telemetry/conftest.py:36` — `in_memory_metric_reader`)
> rather than introducing a project-specific alias. The fixture below is
> a preferences-scoped specialization of that pattern: same name, same
> signature, but additionally rebinds `service._meter` and
> `service._PREFERENCES_PATCH_COUNTER` because the preferences-service
> counter handle is captured at module-import time and the telemetry
> conftest's fixture does not (and should not) know about it. The
> resulting two `in_memory_metric_reader` fixtures live in disjoint
> conftest scopes (the preferences scope overrides the telemetry scope
> for tests under `tests/unit/web/preferences/`); pytest's scoped-fixture
> resolution handles the override cleanly.
>
> Extend `tests/unit/web/preferences/conftest.py` (create if absent):
>
> ```python
> # tests/unit/web/preferences/conftest.py
> from collections.abc import Iterator
> from opentelemetry.sdk.metrics import MeterProvider
> from opentelemetry.sdk.metrics.export import InMemoryMetricReader
> import pytest
>
> from elspeth.web.preferences import service as _service_module
>
>
> @pytest.fixture
> def in_memory_metric_reader(monkeypatch: pytest.MonkeyPatch) -> Iterator[InMemoryMetricReader]:
>     """Rebind ``service._meter`` + ``_PREFERENCES_PATCH_COUNTER`` to a
>     fresh MeterProvider with an InMemoryMetricReader attached.
>
>     Naming parity with ``tests/unit/telemetry/conftest.py:36`` (R2-13,
>     2026-05-19). This is the preferences-scoped specialization: same
>     fixture name, additionally rebinds the module-level counter handle
>     captured at import time.
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
>     - ``tests/unit/telemetry/conftest.py:36`` (canonical
>       ``in_memory_metric_reader`` for the telemetry module — the
>       fixture below mirrors it and adds the counter-handle rebind).
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
>     try:
>         yield reader
>     finally:
>         provider.shutdown()
>         reader.shutdown()
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
> below MUST take `in_memory_metric_reader` as a fixture argument and
> call `_last_patch_counter_attributes(in_memory_metric_reader)` —
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
    service, in_memory_metric_reader
):
    """Panel S1 preservation: `_PREFERENCES_PATCH_COUNTER` emits with the
    full label set including the new `tutorial_changed` label.

    Verifies both that the existing `wrote_row` label survived the additive
    diff AND that the new `tutorial_changed` label was added. Uses the
    ``in_memory_metric_reader`` fixture (see conftest.py — Step 1
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
    attrs = _last_patch_counter_attributes(in_memory_metric_reader)
    assert attrs["mode_changed"] is False
    assert attrs["banner_dismissed"] is False
    assert attrs["wrote_row"] is True  # Phase 1A label preserved.
    assert attrs["tutorial_changed"] is True  # Phase 4 label extension.


def test_patch_without_tutorial_emits_counter_with_tutorial_changed_false(
    service, in_memory_metric_reader
):
    """Counter-label disaggregation: a non-tutorial PATCH still emits the
    counter but with `tutorial_changed=False`."""
    asyncio.run(
        service.update_composer_preferences(
            "alice-counter-mode",
            UpdateComposerPreferencesRequest(default_mode="freeform"),
        )
    )
    attrs = _last_patch_counter_attributes(in_memory_metric_reader)
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
        field_name="tutorial_completed_at",
    )

return ComposerPreferences(
    default_mode=mode,
    banner_dismissed_at=row.banner_dismissed_at,
    tutorial_completed_at=tutorial_completed_at,  # NEW field on response model.
    updated_at=row.updated_at,
)
```

**Update `CorruptPreferencesError` to parameterise on field name (Reality
finding R2-12, 2026-05-19).** The live class's `__init__` at
`src/elspeth/web/preferences/service.py:104-107` hardcodes the message
`"… invalid default_composer_mode=…"`. Raising it for a
`tutorial_completed_at` corruption then prints a misleading
"default_composer_mode" reference. Extend the signature to accept a
`field_name` kwarg (default keeps Phase 1A backwards compatibility for the
existing call sites — both Phase 1A raises pass `default_composer_mode`
explicitly so this is a structural change with zero behaviour change at
the legacy sites):

```python
# === Phase 4 additive delta — CorruptPreferencesError signature ===
# Existing __init__ message hardcodes `default_composer_mode`; extend so
# the field name is structural rather than baked into the format string.
class CorruptPreferencesError(RuntimeError):
    def __init__(
        self,
        user_id: str,
        bad_value: object,
        *,
        field_name: str = "default_composer_mode",
    ) -> None:
        super().__init__(
            f"user_preferences row for {user_id!r} has invalid "
            f"{field_name}={bad_value!r}"
        )
        self.user_id = user_id
        self.bad_value = bad_value
        self.field_name = field_name
```

Update the existing Phase 1A raise sites (line ~189, line ~316) to pass
`field_name="default_composer_mode"` explicitly so the message remains
identical for those call sites. The Phase 4 use packs the bad value as
`{"tutorial_completed_at": raw_tutorial}` AND passes
`field_name="tutorial_completed_at"`, so the message and the structural
attribute both name the corrupt column:

```python
raise CorruptPreferencesError(
    user_id=user_id,
    bad_value={"tutorial_completed_at": raw_tutorial},
    field_name="tutorial_completed_at",
)
```

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

**Cache directory:** resolved from `WebSettings.tutorial_cache_dir`, which
defaults to `<data_dir>/tutorial_cache/` (i.e. `data/tutorial_cache/` in a
dev checkout, since `data_dir` defaults to `Path("data")`). Operators can
override by setting `ELSPETH_WEB__TUTORIAL_CACHE_DIR` to a fully-qualified
path. Tests construct the cache with an explicit `cache_dir` argument
pointing into `tmp_path`. The cache module does **not** read any env vars
directly — the path flows in via the validated `WebSettings` field, per
the config-contracts framework.

**File naming:** `<sha256_hex>.json` where the hex is the SHA-256 of
`f"{canonical_prompt}:{model_id}"`. The plain canonical prompt and model
are also stored inside the JSON for diagnostic visibility (an operator
inspecting a file should be able to confirm what it caches without
recomputing the hash).

**Tier classification — server-generated content cache.** The tutorial
cache stores deterministic LLM output content. **Operationally** the cache
follows Tier-1 rules (crash on corruption with file path + parse error
chained via `from`; miss on absence; never fall back to a live LLM call on
cache content corruption). **Conceptually** the data is server-generated
content derived from an external LLM call, not Tier-1 "our data" in the
CLAUDE.md sense. The crash-on-corrupt invariant exists not because we own
the data but because we wrote the file and corruption indicates a fault we
must surface, not paper over. Future caches that store LLM-derived content
should reuse this framing rather than expanding the Tier-1 envelope.

**Operational guarantees** (corruption discipline; the binding contract
for the cache module):

- A file present → must parse via Pydantic. Parse failure → crash. No
  fallback to a live run: that would mask corruption.
- A file absent → cache miss (legitimate).
- A file's recorded `(canonical_prompt, model_id)` must match what we expected
  for the given key — if not, the file is in the wrong location and we crash.
  (This guards against a misconfigured operator copying files around.)

**Cache-population path in production: the warm-cache CLI.** This module
exposes `TutorialCache.store(...)` as its sole write surface. In
production the cache is populated by `elspeth tutorial warm-cache` (see
§"Cache warming (post-deploy, post-restart)" above), which is a thin
CLI wrapper over the same canonical-seed run-path that an interactive
tutorial user triggers — it ends in a `TutorialCache.store(...)` call.
The wrapper itself is owned by Task 5 (cache module) plus a small CLI
entry-point; no separate cache-only code path exists. Operators ship a
new deployment, delete the sessions DB, restart the service, and run
the warm-cache command before opening the deployment to users.

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
    TutorialCacheCorruptError,
    TutorialCacheEntry,
)


@pytest.fixture
def cache_dir(tmp_path: Path) -> Path:
    d = tmp_path / "tutorial_cache"
    d.mkdir()
    return d


@pytest.fixture
def cache(cache_dir: Path) -> TutorialCache:
    return TutorialCache(cache_dir=cache_dir)


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
    with pytest.raises(TutorialCacheCorruptError, match="not valid JSON"):
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
    with pytest.raises(TutorialCacheCorruptError, match="prompt mismatch"):
        cache.lookup(CANONICAL_SEED_PROMPT, "claude-opus-4-7")


def test_store_leaves_exactly_one_json_file(
    cache: TutorialCache, cache_dir: Path
) -> None:
    """After a clean write, the directory holds exactly one .json file (no tempfiles).

    This is the *clean-path* half of the atomicity contract: write goes through
    a tempfile + rename, so no `.tmp` should survive a successful store. The
    *crash-path* half — that a failed rename leaves zero .json files — is tested
    separately in `test_store_atomic_under_oserror`.
    """
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


def test_store_atomic_under_oserror(
    cache: TutorialCache, cache_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Atomicity contract: if os.replace fails mid-write, no .json file is left behind.

    Simulates a crash between the tempfile write and the rename. The store()
    call must propagate the OSError (no defensive swallow per CLAUDE.md), and
    the cache directory must contain zero .json files afterwards — the
    half-written tempfile, if it exists, is never observable as a cache entry.
    The .tmp file may or may not remain (cleanup is best-effort); the binding
    contract is "no observable .json".
    """
    entry = TutorialCacheEntry(
        canonical_prompt=CANONICAL_SEED_PROMPT,
        model_id="claude-opus-4-7",
        cached_at=datetime(2026, 5, 15, tzinfo=UTC),
        rows=[{"url": "example.gov.au"}],
        source_data_hash="hash",
        llm_call_count=0,
        pipeline_yaml=_CANONICAL_PIPELINE_YAML,
    )

    def boom(*args: object, **kwargs: object) -> None:
        raise OSError("simulated rename failure")

    monkeypatch.setattr("os.replace", boom)

    with pytest.raises(OSError, match="simulated rename failure"):
        cache.store(entry)

    # No .json file should be left after the failed rename.
    assert list(cache_dir.glob("*.json")) == []


def test_tutorial_cache_dir_defaults_to_data_dir_subdir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Default tutorial_cache_dir lives under data_dir, no extra env var required.

    Dev environments without staging env vars must produce a working
    cache directory; the previous hardcoded absolute-path default broke
    devs (root-owned, doesn't exist on dev hosts). This test asserts the P8
    resolution: ``WebSettings.tutorial_cache_dir`` is a ``Path | None``
    field whose ``model_validator`` defaults the value to
    ``data_dir / "tutorial_cache"`` when unset, and the cache module
    accepts the resolved path via constructor injection.
    """
    # Import inside the test so the module-load order doesn't matter for
    # the env-var manipulation below.
    from elspeth.web.config import WebSettings
    from elspeth.web.preferences.tutorial_cache import TutorialCache

    monkeypatch.setenv("ELSPETH_WEB__DATA_DIR", str(tmp_path))
    monkeypatch.delenv("ELSPETH_WEB__TUTORIAL_CACHE_DIR", raising=False)

    settings = WebSettings()

    assert settings.tutorial_cache_dir == tmp_path / "tutorial_cache"
    # Constructor smoke: the cache module accepts the resolved path
    # without raising on the missing-directory case (lookup returns None
    # on a missing-file miss; store() will mkdir parents=True on write).
    cache = TutorialCache(cache_dir=settings.tutorial_cache_dir)
    assert cache.lookup(CANONICAL_SEED_PROMPT, "claude-opus-4-7") is None
```

- [ ] **Step 2: Run to fail.** `.venv/bin/python -m pytest tests/unit/web/preferences/test_tutorial_cache.py -v` → FAIL (`ModuleNotFoundError: tutorial_cache`).

- [ ] **Step 3: Implement the cache module.**

Create `src/elspeth/web/preferences/tutorial_cache.py`:

```python
"""Flat-file tutorial-seed run cache. Absence = miss; corruption = crash.

Key: ``SHA-256(canonical_prompt + ":" + "{composer_model}:{transform_model}")``.

``model_id`` is a compound key produced by ``_model_id_for_pipeline`` —
``"{composer_model}:{transform_model}"`` — because output is sensitive to
both the composer LLM that shapes the pipeline AND the in-pipeline
``llm_rate`` transform model that performs the rating. Either changing
invalidates the cache. The cache module itself treats ``model_id`` as an
opaque string; the compound shape is the caller's contract (see
``_model_id_for_pipeline`` in the run-path module).

Invalidate the cache by deleting the directory.
"""

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


class TutorialCacheCorruptError(RuntimeError):
    """Raised when a present cache file cannot be parsed or fails its
    self-consistency check (prompt/model recorded in the file does not
    match the lookup key).

    Reality finding R2-11 (2026-05-19): the original draft raised bare
    ``RuntimeError`` strings; CLAUDE.md "offensive programming" wants a
    *named* exception so the caller's exception handlers can match by
    class. Subclasses ``RuntimeError`` so existing
    ``except RuntimeError`` chains keep working during the transition
    (there are none today; this is forward-fit headroom).

    Attributes:
      path: filesystem path of the offending cache file.
      parse_error: the chained exception that surfaced the corruption.
    """

    def __init__(self, path: Path, reason: str) -> None:
        super().__init__(f"tutorial cache file {path}: {reason}")
        self.path = path
        self.reason = reason


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
    """Flat-file cache for canonical-seed run outputs.

    The cache directory is injected via ``cache_dir``; the module does
    **not** read any environment variables. Callers resolve the path via
    the validated ``WebSettings.tutorial_cache_dir`` field (the
    app-composition site does this in Task 6) so that config flows through
    the config-contracts framework rather than ad-hoc env lookups.
    """

    def __init__(self, *, cache_dir: Path) -> None:
        self._dir = cache_dir

    def lookup(self, canonical_prompt: str, model_id: str) -> TutorialCacheEntry | None:
        """Return cached entry, or None on miss. Crashes on corruption.

        All corruption surfaces as ``TutorialCacheCorruptError`` (Reality
        finding R2-11, 2026-05-19) — bare ``RuntimeError`` was replaced by
        the named class so callers can match by type. Every raise chains
        the underlying cause via ``from exc`` (CLAUDE.md offensive
        programming: preserve exception chains).
        """
        key = _compute_key(canonical_prompt, model_id)
        path = self._dir / f"{key}.json"
        if not path.exists():
            return None
        try:
            raw = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise TutorialCacheCorruptError(path, "unreadable") from exc
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise TutorialCacheCorruptError(path, "not valid JSON") from exc
        try:
            entry = TutorialCacheEntry.model_validate(data)
        except ValidationError as exc:
            raise TutorialCacheCorruptError(
                path, "does not match expected shape"
            ) from exc
        if entry.canonical_prompt != canonical_prompt or entry.model_id != model_id:
            raise TutorialCacheCorruptError(
                path,
                f"prompt mismatch: file recorded "
                f"({entry.canonical_prompt!r}, {entry.model_id!r}) "
                f"but lookup was for ({canonical_prompt!r}, {model_id!r})",
            )
        return entry

    def store(self, entry: TutorialCacheEntry) -> None:
        """Persist the entry atomically (tempfile + os.replace).

        No defensive ``except OSError: pass`` on the cleanup path (N-R2-2,
        2026-05-19). Per CLAUDE.md offensive programming, a tempfile
        cleanup error indicates filesystem state we do not understand;
        surface it rather than swallow. The outer ``raise`` re-raises the
        original write/replace failure; if the unlink itself fails, the
        ``OSError`` propagates instead and the caller sees the deeper
        problem.
        """
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
            # CLAUDE.md offensive programming: do NOT swallow tempfile
            # cleanup errors. The unlink may fail (e.g., tempfile already
            # removed by another process); let the OSError propagate so
            # the operator sees the unexpected filesystem state. The
            # original exception is preserved by the bare ``raise`` below
            # only when the unlink succeeds.
            tmp_path.unlink(missing_ok=True)
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
# Directory is resolved from WebSettings.tutorial_cache_dir, which defaults
# to ``<data_dir>/tutorial_cache/`` (see field declaration on WebSettings
# below). No env-var lookup happens here or inside TutorialCache — the
# path flows through validated config only, per config-contracts.
# Operators invalidate by deleting the directory.
app.state.tutorial_cache = TutorialCache(
    cache_dir=settings.tutorial_cache_dir
)
```

Add `tutorial_cache_dir: Path | None = Field(default=None)` to `WebSettings`
with a `model_validator(mode="after")` that defaults the field to
`data_dir / "tutorial_cache"` when unset. `WebSettings` is `frozen=True`
(see `model_config = ConfigDict(frozen=True)` near the top of
`src/elspeth/web/config.py`), so the validator must assign via
`object.__setattr__`. Operators override by setting
`ELSPETH_WEB__TUTORIAL_CACHE_DIR` to a fully-qualified path; the absent
case must **never** fall back to a hardcoded absolute path — the only
baseline is `data_dir`, which is itself a validated,
resolved-to-absolute field.

```python
from pydantic import Field, model_validator

class WebSettings(BaseSettings):
    # ... existing fields ...
    tutorial_cache_dir: Path | None = Field(default=None)

    @model_validator(mode="after")
    def _default_tutorial_cache_dir(self) -> "WebSettings":
        # Resolve relative to the validated ``data_dir`` (which has
        # already been .expanduser().resolve()'d by the existing
        # _normalize_paths field validator). Never fall back to a
        # hardcoded absolute path — the P8 fix.
        if self.tutorial_cache_dir is None:
            object.__setattr__(
                self, "tutorial_cache_dir", self.data_dir / "tutorial_cache"
            )
        return self
```

Startup write-permission check (offensive programming): on app boot,
after `WebSettings()` is constructed and before `TutorialCache` is wired,
the app-composition code must verify the directory is creatable and
writable, and crash with `RuntimeError` naming the resolved path and the
service OS user if not. The check belongs at the app-composition site
(it touches the filesystem, which a Pydantic validator must not do), not
inside `TutorialCache`.

- [ ] **Step 3: Wire the Landscape repository dependency providers.**

Per operator decision CR-3 (2026-05-19), Phase 4's new Landscape access
flows through **FastAPI `Depends`-injected repositories**, NOT through
`app.state.landscape` or `app.state.run_lifecycle_repo` shims. Two
repository surfaces are needed:

- `RunLifecycleRepository` (read side, **already exists** at
  `src/elspeth/core/landscape/run_lifecycle_repository.py:82`). Task 7.2
  reads `runs_table` rows through this repository's existing
  `get_run(landscape_run_id)` method.
- `LandscapeWriteRepository` (write side, **NEW class** added by
  Task 7.0). Task 7's `_replay_cached_content_to_landscape` synthesises
  a new Landscape entry through this repository's `record_synthesised_run`
  method. The split mirrors the existing repository pattern in the
  module (cf. `auth_audit_repository.py`, `query_repository.py`,
  `data_flow_repository.py`, `execution_repository.py`).

Add the dependency providers alongside the cache wiring. Their bodies
construct the repository against the project's standard Landscape
engine handle (confirm exact construction during recon — likely
`request.app.state.landscape_engine` or a module-level singleton; if
the project does not already expose a Landscape engine on app state,
this is an addition that belongs to Task 7.0):

```python
from elspeth.core.landscape.run_lifecycle_repository import RunLifecycleRepository
from elspeth.core.landscape.write_repository import LandscapeWriteRepository

# Phase 4A.6 — repository dependency providers. NO `app.state.landscape`
# or `app.state.run_lifecycle_repo` attribute shim is introduced; the
# repositories are constructed per-request through FastAPI's `Depends`
# graph so they pick up the right engine handle without leaking a
# global `app.state` surface (per operator decision CR-3, 2026-05-19).

def get_run_lifecycle_repo(request: Request) -> RunLifecycleRepository:
    """FastAPI dependency: read-side Landscape repository."""
    return RunLifecycleRepository(engine=request.app.state.landscape_engine)

def get_landscape_write_repo(request: Request) -> LandscapeWriteRepository:
    """FastAPI dependency: write-side Landscape repository (new in Task 7.0)."""
    return LandscapeWriteRepository(engine=request.app.state.landscape_engine)
```

Routes that previously read from `app.state.run_lifecycle_repo` (the
audit-story route in Task 7.2) and the route that invokes
`_replay_cached_content_to_landscape` (Task 7.1) take the repository
via `Depends(get_run_lifecycle_repo)` / `Depends(get_landscape_write_repo)`
on the route handler signature. No bare `request.app.state.landscape`
or `request.app.state.run_lifecycle_repo` access is permitted anywhere
in the Phase 4 surface.

- [ ] **Step 4: Smoke-test that the app still starts.**

```bash
.venv/bin/python -m pytest tests/integration/web/ -v -x
```

Expected: PASS — no existing integration tests should break.

- [ ] **Step 5: Commit.**

```bash
git add <app-file>
git commit -m "feat(web): wire TutorialCache + Landscape repo deps onto app composition (Phase 4A.6)"
```

## Task 7.0: Schema — extend `runs_table` with audit-story columns

**Files:**
- Modify: `src/elspeth/core/landscape/schema.py` — add three columns to `runs_table`.
- Modify: `src/elspeth/contracts/audit.py` — extend the `Run` dataclass
  (`audit.py:72`) with three new optional fields.
- Create: `src/elspeth/core/landscape/write_repository.py` — new
  `LandscapeWriteRepository` class with a `record_synthesised_run`
  method. Mirrors the existing repository pattern
  (`auth_audit_repository.py`, `query_repository.py`,
  `data_flow_repository.py`, `execution_repository.py`,
  `run_lifecycle_repository.py`).
- Modify: the Landscape write path that records a completed run (recon
  below) — populate the new columns.
- Modify: the engine's run-scope accumulator state (recon below) — add an
  `llm_call_count` integer counter incremented on every LLM transform
  call within a run.
- Create: `tests/unit/core/landscape/test_runs_table_audit_story_columns.py` — column-presence + write-side fixture tests.

**Why this is a separate task (and why it lands before Task 7).** The
audit-story endpoint added in Task 7.2 reads several fields from a
Landscape run row. Three of those fields are added to `runs_table` by
this task: `llm_call_count`, `seeded_from_cache`, `cache_key`. Two others
the response surfaces (`source_data_hash`, `plugin_versions`) already
exist at row/node level — `rows_table.source_data_hash` at
`schema.py:168` and `nodes_table.plugin_version` at `schema.py:120`
(verified 2026-05-19) — and Task 7.2's service aggregates them at query
time rather than denormalising them onto `runs_table` (Systems finding
R2-S4, 2026-05-19). The existing `started_at` column at `schema.py:65`
is reused unchanged (Reality finding R2-5, 2026-05-19).

Verified against the live schema (`src/elspeth/core/landscape/schema.py`
as of 2026-05-19), **the three new columns do not exist on `runs_table`
today**. The table has `started_at`, `config_hash`, and other run-config
metadata, but no LLM-call counter and no cache-replay markers.

**Final column list (R2-S4 — query-time aggregation honours
no-denormalization):**

| Column | Source | Notes |
|--------|--------|-------|
| `llm_call_count` (NEW) | run-scope accumulator (this task adds it) | Nullable; NULL on pre-Phase-4 rows and on live runs until the accumulator lands. See R2-S3 below. |
| `seeded_from_cache` (NEW) | written by Task 7's `_replay_cached_content_to_landscape`; False on live runs | NOT NULL with `server_default=text("0")` (N-R2-4, 2026-05-19). |
| `cache_key` (NEW) | written by Task 7's replay path; NULL on live runs | Nullable. |
| `started_at` (REUSED) | existing column at `schema.py:65` | No new code; Task 7.2 reads the existing column. |
| `source_data_hash` (REUSED — aggregated) | Task 7.2 aggregates from `rows_table.source_data_hash` | Not added to `runs_table` (R2-S4). |
| `plugin_versions` (REUSED — aggregated) | Task 7.2 aggregates from `nodes_table.plugin_version` joined per-run | Not added to `runs_table` (R2-S4). |

Task 7's replay path (`_replay_cached_content_to_landscape`) and Task 7.2's
read path both assume these columns. Without Task 7.0, Task 7 will write
metadata that has no column to land in, and Task 7.2's
`audit_row.llm_call_count` access will raise `AttributeError` at runtime.

**`llm_call_count` accumulator (Systems finding R2-S3, 2026-05-19).** The
field name `llm_call_count` does not exist anywhere in `src/elspeth/`
today; the original draft said "grep confirms the field exists" but it
does not. Adding the column without populating it would leave the
audit-story endpoint reading a permanently-NULL value. This task
therefore introduces both the column AND the accumulator:

1. **Recon (Step 1 below) identifies the run-scope object** that owns
   per-run metadata during execution (likely an attribute on
   `Orchestrator` in `src/elspeth/engine/orchestrator.py`, or a counter
   on a `RunRecord` analogue in `core/landscape/`). The accumulator is
   an integer counter incremented at the LLM-transform call site.
2. **Operator-approved write strategy (2026-05-19):** populate
   `llm_call_count` from the cache entry only for cache-replay runs.
   Live (non-tutorial) runs write `NULL` until a sibling phase adds
   general LLM-call counting across all run types. The column is
   therefore **nullable** — pre-Phase-4 rows have NULL, post-Phase-4
   live runs have NULL, only cache-replay runs (and any later
   accumulator wiring) carry a non-NULL count.
3. The audit-story endpoint surfaces NULL → 0 only when the
   request-time run path was a cache replay (the cache entry's
   `cache_seeding_llm_call_count` is what the user sees; the run's
   own `llm_call_count` is 0 by construction); for a live run a NULL
   here is honest — the system didn't count, and we don't fabricate.

**`LandscapeWriteRepository`** is also added by this task because the
columns it populates and the read columns Task 7.2 surfaces are paired —
adding the columns without the write surface ships dead schema. The new
class lives at `src/elspeth/core/landscape/write_repository.py` and
exposes a single async method initially:

```python
async def record_synthesised_run(
    self,
    *,
    session_id: str,
    user_id: str,
    pipeline_yaml: str,
    rows: list[Mapping[str, Any]],
    source_data_hash: str,            # written to rows_table per-row, not runs_table
    llm_call_count: int,              # written to the NEW runs_table column (0 for replay)
    plugin_versions: Mapping[str, str],  # written to nodes_table.plugin_version per-node
    started_at: datetime,
    metadata: Mapping[str, Any],
) -> str:
    """Insert a synthesised `runs_table` row for a cache-replay plus the
    matching `rows_table` and `nodes_table` rows so query-time
    aggregation (Task 7.2 audit-story) can resolve
    `source_data_hash` (from rows) and `plugin_versions` (from nodes)
    without denormalisation (R2-S4, 2026-05-19).

    `metadata` carries `seeded_from_cache` and `cache_key` (Task 7's
    cache-replay marker). Returns the freshly-minted `run_id`.

    Tier-1: all callers MUST supply every argument. No defaults; no
    optional fields silently populated as NULL. Per CLAUDE.md no-
    defensive-programming, the implementation is a multi-statement
    INSERT (runs + rows + nodes) wrapped in a single transaction; bad
    callers crash at the SQL constraint check.
    """
```

**Operator action required.** Per the project's no-Alembic policy (memory:
`project_db_migration_policy`, `project_phase9_sqlite_only`), schema
changes are applied by deleting the existing audit DB and letting the
caretaker re-bootstrap from `schema.py`. **The operator must delete the
deployment's audit DB at `<ELSPETH_WEB__DATA_DIR>/runs/audit.db` (and any
per-eval audit DBs) after this task lands, before the next pipeline
run.** This is the established
DB-migration pattern, but its scope here is the full Landscape audit DB —
surface this to the operator and pause for confirmation before merging
Task 7.0.

**Trust tier.** The three new columns are Tier-1 (Landscape data). Their
write-side population (in Task 7's modified run path and in the
existing run-completion path) follows the operator-approved nullability
rules above:

- `seeded_from_cache` — NOT NULL; live runs write False, cache-replay
  writes True. `server_default=text("0")` (SQLite) covers any path
  that omits the column explicitly.
- `cache_key` — nullable; NULL for live runs, the SHA-256 hex for
  replays.
- `llm_call_count` — nullable; NULL for live runs until the general
  counting accumulator lands, 0 for cache-replay runs (no live LLM
  calls were made).

The aggregated fields (`source_data_hash`, `plugin_versions`) are
already Tier-1 in their source tables (`rows_table` and `nodes_table`
respectively); the audit-story endpoint reads them via JOIN, not via
a duplicate column on `runs_table`.

- [ ] **Step 1: Reconnaissance — identify the live Landscape run-completion write site AND the LLM-call accumulator landing site.**

```bash
grep -rn "runs_table.*insert\|INSERT.*INTO.*runs\|record_run_completion\|persist_run\b" \
  src/elspeth/core/landscape/ --include="*.py" | grep -v __pycache__ | head -20
grep -rn "llm_call_count\|llm_calls_made\|llm.*counter" \
  src/elspeth/engine/ src/elspeth/plugins/transforms/ --include="*.py" \
  | grep -v __pycache__ | head -20
```

Identify the function that inserts a row into `runs_table` at the end of a
successful pipeline run. Confirm the shape of the data it receives:

1. **`llm_call_count` accumulator landing site (R2-S3, 2026-05-19).** The
   field name does not exist anywhere in `src/elspeth/` today; this task
   is its first appearance. Identify where in the engine the accumulator
   should live — likely an attribute on `Orchestrator`
   (`src/elspeth/engine/orchestrator.py`) or a counter on the `RunRecord`
   analogue in `core/landscape/`. The accumulator is incremented at the
   LLM-transform call site (likely
   `src/elspeth/plugins/transforms/llm_rate.py` or its base class). For
   Phase 4, the live-run accumulator may be deferred — but the recon
   must identify the future landing site so the column's NULL semantics
   are honest (NULL = "not counted" — see operator-approved strategy
   above).
2. Where does `source_data_hash` come from? The live `rows_table` carries
   this at row level (`schema.py:168`, confirmed 2026-05-19). Task 7.2
   aggregates from rows; the run-completion write path does NOT
   populate a run-level `source_data_hash` column (R2-S4).
3. `started_at` (run-start timestamp): **reused, not duplicated**. Confirmed
   2026-05-19 to exist already at `schema.py:65`. Task 7.2's response model
   surfaces this as `started_at`; the no-new-column decision is final.
4. `plugin_versions` — the live `nodes_table` carries this at node level
   (`schema.py:120`, confirmed 2026-05-19). Task 7.2 aggregates from
   nodes; the run-completion write path does NOT populate a run-level
   `plugin_versions` column (R2-S4).
5. `seeded_from_cache` / `cache_key` — written by Task 7's
   `_replay_cached_content_to_landscape`; on a normal live run they are
   `False` / `None`. Operator decision (2026-05-19): nullable-but-Tier-1
   with `server_default=text("0")` for `seeded_from_cache` (N-R2-4) so
   any existing INSERT path that omits the column continues to work.

Write the findings into the commit body. If the accumulator landing site
turns out to require non-trivial engine plumbing, surface that to the
operator before proceeding; the operator-approved strategy permits NULL
for live runs in Phase 4 (the column lands; the accumulator follows in a
sibling phase).

- [ ] **Step 2: Write the failing test.**

Create `tests/unit/core/landscape/test_runs_table_audit_story_columns.py`:

```python
"""Column-presence and write-side smoke tests for Task 7.0 additions."""

from __future__ import annotations

from elspeth.core.landscape.schema import runs_table


def test_runs_table_has_audit_story_columns() -> None:
    """The three new Phase 4.7.2 audit-story columns are present on runs_table.

    Note: `started_at` is the run-start timestamp and already exists on
    `runs_table` (see `schema.py:65`). The audit-story response reuses it
    rather than introducing a duplicate `started_at`-alias column (Reality
    finding R2-5, 2026-05-19). `source_data_hash` and `plugin_versions`
    are aggregated by Task 7.2's audit-story service from `rows_table`
    and `nodes_table` respectively, not denormalised onto `runs_table`
    (Systems finding R2-S4, 2026-05-19).
    """
    expected_new = {
        "llm_call_count",
        "seeded_from_cache",
        "cache_key",
    }
    actual = {col.name for col in runs_table.columns}
    missing = expected_new - actual
    assert not missing, f"Missing columns on runs_table: {missing}"
    # Confirm `started_at` still present (reuse-not-rename invariant).
    assert "started_at" in actual, "started_at must remain on runs_table"
    # Confirm we did NOT denormalise the per-row / per-node fields.
    assert "source_data_hash" not in actual, (
        "source_data_hash must remain at rows-table level (R2-S4)"
    )
    assert "plugin_versions" not in actual, (
        "plugin_versions must remain at nodes-table level (R2-S4)"
    )


def test_llm_call_count_is_nullable_integer() -> None:
    """Operator-approved Phase 4 semantics (R2-S3, 2026-05-19): the column
    is nullable until the general accumulator lands in a sibling phase.

    NULL means "not counted" (live runs in Phase 4). Cache-replay runs
    write 0 (no live LLM calls). A future sibling phase will populate
    live-run rows with the actual count; until then NULL is honest.
    """
    col = runs_table.c.llm_call_count
    assert col.nullable, "llm_call_count is nullable in Phase 4 (R2-S3)"


def test_seeded_from_cache_has_server_default() -> None:
    """N-R2-4 (2026-05-19): existing INSERT paths that omit the column
    must continue to work — `server_default=text("0")` provides the
    fallback. The Python-side `default=False` covers ORM constructs;
    the server-side default covers raw INSERTs.
    """
    col = runs_table.c.seeded_from_cache
    assert not col.nullable, "seeded_from_cache must be NOT NULL"
    assert col.server_default is not None, (
        "seeded_from_cache must declare server_default (N-R2-4)"
    )
```

Expect: FAIL — columns do not exist yet.

- [ ] **Step 3: Add the columns.**

In `src/elspeth/core/landscape/schema.py`, extend `runs_table` (the existing
`Table("runs", ...)` block at lines 61-98) with the three new columns. Place
them after `runtime_val_manifest_json` (current last column at line 97) so
the column-order on existing rows is preserved (relevant for the
delete-and-recreate migration path):

```python
from sqlalchemy import text  # if not already imported

# === Phase 4A.7.0 audit-story columns (added 2026-05-19) ===
# These three new columns back the GET /audit-story endpoint (Phase 4A.7.2).
# `seeded_from_cache` and `cache_key` are populated only by
# `_replay_cached_content_to_landscape` (Task 7); on a normal live run
# `seeded_from_cache=False` (covered by server_default) and `cache_key=NULL`.
# `llm_call_count` is nullable in Phase 4 (R2-S3, operator-approved):
# live runs write NULL until a sibling phase adds general LLM-call
# counting; cache-replay runs write 0.
# `started_at` (the run-start timestamp) is NOT added here — it already
# exists at line 65; Task 7.2 reads the existing column rather than
# introducing a duplicate (Reality finding R2-5, 2026-05-19).
# `source_data_hash` and `plugin_versions` are NOT added here — they
# exist at row/node level (`rows_table.source_data_hash:168`,
# `nodes_table.plugin_version:120`) and Task 7.2 aggregates them at
# query time (Systems finding R2-S4, 2026-05-19).
Column("llm_call_count", Integer, nullable=True),
Column("seeded_from_cache", Boolean, nullable=False, default=False, server_default=text("0")),
Column("cache_key", String(64), nullable=True),
```

Note: `Integer`, `Boolean`, and `text` must be imported from `sqlalchemy`
alongside the existing `Column`, `String`, `DateTime` imports. The
`server_default=text("0")` is the SQLite convention for a boolean
NOT NULL column (N-R2-4, 2026-05-19); SQLAlchemy renders this as
`DEFAULT 0` in the CREATE TABLE statement, which any pre-existing INSERT
path that omits `seeded_from_cache` falls back to.

The `Run` dataclass at `src/elspeth/contracts/audit.py:72` must be
extended with the matching three optional fields so the read path
(Task 7.2) can return them. `started_at` is already on `Run`
(`audit.py:79`) and is reused unchanged — do NOT add a duplicate.
`source_data_hash` and `plugin_versions` are NOT added to `Run` (they
remain on `Row` and `Node` respectively — Task 7.2 aggregates):

```python
# Phase 4A.7.0 — audit-story projection fields (R2-S4 final list).
# Optional with None defaults because pre-Phase-4 rows do not have
# these columns.
llm_call_count: int | None = None       # R2-S3: nullable; NULL for live runs
seeded_from_cache: bool = False
cache_key: str | None = None
```

- [ ] **Step 4: Extend the run-completion write site.**

In the file Step 1 identified, extend the `runs_table` INSERT to populate
the three new columns. For live runs (Phase 4 baseline):

- `llm_call_count`: write `NULL`. The general accumulator lands in a
  sibling phase (R2-S3, operator-approved 2026-05-19). NULL is honest;
  fabricating 0 here would assert "no LLM calls happened" which is
  almost certainly false.
- `seeded_from_cache`: `False` (covered by `server_default=text("0")`
  if the column is omitted; populating explicitly is preferred for
  clarity).
- `cache_key`: `None`.

Note: `started_at` (the run-start timestamp) is already populated by the
existing run-completion path. No new code needed for that column —
Task 7.2's response model just reads the existing column.

`source_data_hash` and `plugin_versions` are NOT populated at run level
(R2-S4). They already exist at row level (`rows_table.source_data_hash`,
populated by the source loader on each row insert) and node level
(`nodes_table.plugin_version`, populated by the node-registration write
path). Task 7.2's audit-story service aggregates them at query time via
JOIN.

For cache-replay runs (the path in Task 7's
`_replay_cached_content_to_landscape`), the values come from the cached
entry plus the replay's own cache-key computation. The replay path:

- `llm_call_count`: write `0` (no live LLM calls were made — the cache
  served the responses; this is not fabrication because the value is a
  fact about *this* run, not an inferred value about a previous run).
- `seeded_from_cache`: `True`.
- `cache_key`: the SHA-256 hex computed by `_compute_key`.

The replay must also insert into `rows_table` (one row per cached row
carrying the cached `source_data_hash`) and `nodes_table` (one row per
node from the cached `pipeline_yaml` plus its plugin version) so
Task 7.2's aggregation surfaces the same hash/version values an auditor
would see on a live run. This multi-table write is the body of
`LandscapeWriteRepository.record_synthesised_run` (transaction-wrapped
INSERTs into `runs_table` + `rows_table` + `nodes_table`).

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
deployment's configured `ELSPETH_WEB__DATA_DIR`):
  <data_dir>/runs/audit.db                   (deployed audit DB)
  examples/*/runs/audit.db                   (example pipelines)
  evals/*/audit.db                           (eval harness)

Confirm with operator. Do NOT proceed to Task 7 until this is acknowledged.
```

- [ ] **Step 7: Commit.**

```bash
git add src/elspeth/core/landscape/schema.py \
        src/elspeth/contracts/audit.py \
        src/elspeth/core/landscape/write_repository.py \
        <run-completion-write-site> \
        tests/unit/core/landscape/test_runs_table_audit_story_columns.py
git commit -m "$(cat <<'EOF'
feat(landscape): add audit-story columns + LandscapeWriteRepository (Phase 4A.7.0)

OPERATOR ACTION: Delete the live Landscape audit DBs before the next
pipeline run. Paths to remove (verify the deployment's configured
`ELSPETH_WEB__DATA_DIR`):
  <data_dir>/runs/audit.db                  (deployed audit DB)
  examples/*/runs/audit.db                  (example pipelines)
  evals/*/audit.db                          (eval harness)

Three new columns on runs_table (R2-S4 final list, 2026-05-19):
  - llm_call_count       (nullable; R2-S3 — NULL on live runs until
                          general accumulator lands)
  - seeded_from_cache    (NOT NULL, server_default=0; N-R2-4)
  - cache_key            (nullable)

source_data_hash and plugin_versions are NOT added at run level —
Task 7.2's audit-story service aggregates them from rows_table and
nodes_table at query time (no denormalisation, R2-S4).

The existing `started_at` column is reused for the audit-story
endpoint's run-start timestamp (no duplicate `started_at`-alias column —
Reality finding R2-5, 2026-05-19).
EOF
)"
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
   successfully AND every row produced a clean rating, calls
   `app.state.tutorial_cache.store(...)` to populate the cache for the
   next user. The cache-write gate is `_is_successful_run(result) AND
   _all_rows_succeeded(result)` — P18's all-rows-succeeded condition
   is now in place; a partial-success run (some rows quarantined or
   carrying `rating_error`) records a Landscape entry for the user
   who ran it but does NOT poison the shared cache.

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
from contextlib import contextmanager
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


# === P18 cache-gate test fixtures (R2-M3, 2026-05-19) ===
# These context managers patch the live run-path so a specific row in
# the canonical-seed result carries either a `rating_error` or a `None`
# rating — exercising the `_all_rows_succeeded` gate without needing
# real LLM failures. The exact patch target depends on Task 7 Step 1
# recon (the function that builds the row dicts from raw LLM output);
# placeholder is `elspeth.web.composer.run_path._execute_pipeline_live`
# but the implementer rebinds during recon.

@contextmanager
def _force_one_row_to_error(app: FastAPI, row_index: int = 0) -> Iterator[MagicMock]:
    """Patch the run-path so the indexed row returns a `rating_error`.

    The other rows return clean ratings. Used by the P18 cache-gate
    tests — a run with any errored row records a Landscape entry but
    does NOT poison the shared cache.

    Recon-dependent: the patch target below is a placeholder. Task 7
    Step 1 identifies the canonical patch site (likely
    ``elspeth.web.composer.run_path._execute_pipeline_live`` or the
    deeper LLM-transform call). Bind the target consistently across
    both `_force_*` helpers below.
    """
    with patch(
        "elspeth.web.composer.run_path._execute_pipeline_live",
        new_callable=AsyncMock,
    ) as mock:
        mock.return_value = {
            "rows": [
                {
                    "url": f"https://example.gov.au/{i}",
                    "rating": None if i == row_index else 5,
                    "rating_error": "simulated 503 from upstream" if i == row_index else None,
                }
                for i in range(5)
            ],
            "source_data_hash": "live-hash-partial",
            "llm_call_count": 4,
            "pipeline_yaml": _CANONICAL_PIPELINE_YAML,
            "run_id": "run-partial-success",
            # Other fields populated as the live shape requires; recon
            # fills these in to match `_execute_pipeline_live`'s real
            # return type.
        }
        yield mock


@contextmanager
def _force_one_row_to_have_null_rating(
    app: FastAPI, row_index: int = 0
) -> Iterator[MagicMock]:
    """Patch the run-path so the indexed row has a `None` rating but
    NO `rating_error`. The absence-of-rating case must still block the
    cache write — the audit trail records what we got; the cache must
    not surface absent data as confident output.
    """
    with patch(
        "elspeth.web.composer.run_path._execute_pipeline_live",
        new_callable=AsyncMock,
    ) as mock:
        mock.return_value = {
            "rows": [
                {
                    "url": f"https://example.gov.au/{i}",
                    "rating": None if i == row_index else 5,
                    "rating_error": None,  # no explicit error — just absent rating
                }
                for i in range(5)
            ],
            "source_data_hash": "live-hash-absent",
            "llm_call_count": 5,
            "pipeline_yaml": _CANONICAL_PIPELINE_YAML,
            "run_id": "run-absent-rating",
        }
        yield mock


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
    app.state.tutorial_cache = TutorialCache(cache_dir=cache_dir)

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


def test_cache_write_skipped_when_any_row_errors(
    app_with_cache: FastAPI, cache_dir: Path
) -> None:
    """P18 — partial-success run records Landscape but does NOT poison the cache.

    A canonical-seed live run where one URL returns 503 produces a row with
    `rating_error` set; the pipeline itself doesn't crash, the Landscape
    entry is recorded normally, the user gets their (degraded) result —
    but the cache MUST remain empty so the next user gets a fresh run
    that might succeed cleanly.
    """
    client = TestClient(app_with_cache)
    # Configure the run-path's fake LLM / fake web_scrape to fail on one URL.
    # The exact rigging mechanism is recon-dependent (whether the test
    # patches `_execute_pipeline_live` or feeds the fake through
    # `app.state.llm_client`); the executor picks the lowest-impact patch
    # point during Step 1.
    with _force_one_row_to_error(app_with_cache):
        response = client.post(
            "/api/sessions/<session_id>/runs",
            json={"<canonical-seed-pipeline-shape>": "..."},
        )
    # Run itself succeeds; the user sees a 200 with a row carrying
    # rating_error.
    assert response.status_code == 200
    body = response.json()
    erroring_rows = [r for r in body["rows"] if r.get("rating_error") is not None]
    assert len(erroring_rows) >= 1
    # Cache write was gated by `_all_rows_succeeded`; the directory
    # remains empty.
    assert list(cache_dir.iterdir()) == []


def test_cache_write_skipped_when_any_row_missing_rating(
    app_with_cache: FastAPI, cache_dir: Path
) -> None:
    """P18 — a row with no `rating_error` but a None rating still blocks cache write.

    The eligibility predicate treats absence-of-rating as a partial
    failure (the audit trail must reflect what we got; the cache must
    not surface absent data as confident output).
    """
    client = TestClient(app_with_cache)
    with _force_one_row_to_have_null_rating(app_with_cache):
        response = client.post(
            "/api/sessions/<session_id>/runs",
            json={"<canonical-seed-pipeline-shape>": "..."},
        )
    assert response.status_code == 200
    assert list(cache_dir.iterdir()) == []
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
            model_id = _model_id_for_pipeline(
                request.app.state.settings, pipeline_state
            )
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
                    write_repo=write_repo,  # Depends-injected; see Task 6 Step 3
                    user=user,
                    session_id=session_id,
                    cache_entry=cache_entry,
                    cache_key=_compute_key(CANONICAL_SEED_PROMPT, model_id),
                )

    # Normal path: run live.
    result = await _execute_pipeline_live(request, user, session_id, pipeline_state)

    # Tutorial-mode + canonical-seed cache populate (on success only).
    # Cache-write eligibility (P18): the run must have produced a clean
    # rating for EVERY row before we are willing to bake it into the
    # cache. A run with row-level LLM failures (one URL returned 503;
    # the transform recorded an error on that row) is "successful" at
    # the run level (the pipeline didn't crash) but is NOT cache-write
    # eligible — caching the degraded output locks the next user into
    # the same degraded experience on every cache hit. See
    # `_all_rows_succeeded` below for the eligibility check.
    if (
        prefs.tutorial_completed_at is None
        and _is_canonical_seed_pipeline(pipeline_state)
        and _is_successful_run(result)
        and _all_rows_succeeded(result)
    ):
        model_id = _model_id_for_pipeline(
            request.app.state.settings, pipeline_state
        )
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
    write_repo: LandscapeWriteRepository,
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

    ``write_repo`` is the ``LandscapeWriteRepository`` introduced by
    Task 7.0; it is FastAPI-``Depends``-injected at the route boundary
    via ``get_landscape_write_repo`` (Task 6 Step 3). No
    ``request.app.state.landscape`` attribute exists in the Phase 4
    surface (operator decision CR-3, 2026-05-19).
    """
    # `record_synthesised_run` is the write surface added by Task 7.0.
    # It writes one runs_table row PLUS the matching rows_table /
    # nodes_table rows so query-time aggregation (R2-S4) resolves
    # source_data_hash (from rows) and plugin_versions (from nodes)
    # the same way it would for a live run.
    #
    # plugin_versions are derived from cache_entry.pipeline_yaml — the
    # replay parses the cached YAML and resolves each plugin's version
    # via the same registry the live run-path uses (recon-dependent —
    # the implementer wires the resolution call here).
    resolved_plugin_versions = _resolve_plugin_versions_from_yaml(
        cache_entry.pipeline_yaml
    )
    new_run_id = await write_repo.record_synthesised_run(
        session_id=session_id,
        user_id=user.user_id,
        pipeline_yaml=cache_entry.pipeline_yaml,
        rows=cache_entry.rows,
        source_data_hash=cache_entry.source_data_hash,
        llm_call_count=0,  # cache replay made no live LLM calls
        plugin_versions=resolved_plugin_versions,
        started_at=datetime.now(UTC),
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


def _all_rows_succeeded(result: PipelineRunResult) -> bool:
    """Cache-write eligibility gate (P18).

    Returns True iff EVERY row in the run produced a clean LLM rating
    (no rating_error, no missing rating field). A run with any errored
    row is NOT eligible for cache write — caching it would bake the
    degraded experience into the cache for every subsequent user who
    hits the canonical seed.

    Rationale: `_is_successful_run` answers "did the pipeline complete
    without crashing"; that is the level at which we record a Landscape
    entry and return a response to the frontend, and we want that to
    stay permissive (a partial-success run still produces a real
    audit trail for the user who ran it). But the cache is shared
    across all future tutorial users hitting the same canonical seed,
    and the cache-miss path on a future user would still execute fresh
    and might succeed — locking in a partial-failure cache would make
    that less likely. Therefore the cache-write eligibility is
    strictly stricter than run-success: ALL rows must have a clean
    rating.

    Operates on the same `result` shape that `_execute_pipeline_live`
    returns. The exact field names depend on recon (the row-level
    error key may be `rating_error`, `error`, `llm_error`, or carried
    on a status enum); the executor confirms during Task 7 Step 1
    and binds the predicate body to the live shape.

    CLAUDE.md offensive programming (CR-4 / R2-M5, 2026-05-19): direct
    key access on each row dict, NOT `row.get(key)`. A missing key here
    means an upstream transform produced a malformed row dict — that is
    a system-code bug we want to surface as ``KeyError`` rather than
    silently treat as "row succeeded". The Tier-2 rule is "no coercion
    at transform/sink level"; the cache-eligibility predicate sits
    structurally at sink level and follows the same rule. The row dicts
    are produced by the run-path's terminal node, which guarantees the
    `rating_error` and `rating` keys are always present (`rating_error`
    may be `None`, `rating` may be `None` — but the *keys* are present).
    """
    for row in result["rows"]:
        # Direct key access — CLAUDE.md offensive programming (CR-4).
        # KeyError here indicates an upstream transform/sink bug, NOT a
        # row to silently skip. Recon confirms the canonical key names
        # against the live terminal-node output shape.
        if row["rating_error"] is not None:
            return False
        if row["rating"] is None:
            # Missing rating = partial failure even without an explicit
            # error key — the cache must not surface absent data.
            return False
    return True
```

The `_is_canonical_seed_pipeline`, `_model_id_for_pipeline`,
`_is_successful_run`, `_all_rows_succeeded`, and
`_execute_pipeline_live` helpers are defined in the same file. The
first, third, fourth, and fifth depend on recon; the executor fills
them in based on the actual run-path internals.
`_replay_cached_content_to_landscape` replaces the old
`_cached_entry_to_run_response` — the substantive change is that it writes
a real Landscape entry under the current session rather than synthesising a
response dict containing a foreign `run_id`.

`_model_id_for_pipeline` is **not** recon-dependent: its shape is fixed by
the cache-key contract. Output is sensitive to **both** the composer LLM
model (which interprets the prompt and shapes the pipeline) **and** the
in-pipeline transform model (which executes the actual rating in `llm_rate`).
Either changing must invalidate the cache, so the helper returns a compound
key spanning both:

```python
from elspeth.web.composer.state import CompositionState

def _model_id_for_pipeline(
    settings: WebSettings, pipeline_state: CompositionState
) -> str:
    """Compose the cache-key model_id from both models that affect output.

    Output for the canonical seed is sensitive to two distinct models:

    1. ``settings.composer_model`` — the LLM that interprets the seed prompt
       and emits the pipeline shape. Configured at service-startup via
       ``WebSettings``; same for every request in a given deployment.
    2. The ``llm_rate`` transform's ``model`` option inside ``pipeline_state``
       — the LLM that performs the rating step at runtime. Per-pipeline, lives
       in the YAML the composer emitted.

    Either changing must invalidate the cache (different (1) means a
    different pipeline shape; different (2) means different ratings even for
    the same shape). Format: ``"{composer_model}:{transform_model}"``.

    Tier-1 crash-on-absence: every canonical-seed pipeline must carry an
    ``llm_rate`` transform with a populated ``model`` option. If extraction
    fails (no ``llm_rate`` transform, or its ``model`` option is absent),
    the helper raises — that is a pipeline-shape invariant violation, not a
    recoverable cache miss. The exact extraction code is recon-dependent
    (it traverses ``pipeline_state`` per the project's pipeline-state
    representation); the contract is the compound key shape, not the
    traversal mechanics.

    Type note: ``CompositionState`` is the live class name at
    ``src/elspeth/web/composer/state.py:1654`` (Reality finding R2-15,
    2026-05-19 — an earlier draft referred to ``PipelineState``, which is
    not a class in the live codebase).
    """
    transform_model = _extract_llm_rate_model(pipeline_state)  # crash on absence
    return f"{settings.composer_model}:{transform_model}"
```

The two call sites above pass `request.app.state.settings` alongside
`pipeline_state` when invoking the helper.

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

## Task 9: Warm-cache CLI — `elspeth tutorial warm-cache`

**Files:**
- Create: `src/elspeth/cli/tutorial.py` — CLI subcommand module.
- Modify: the top-level CLI registration (likely `src/elspeth/cli.py` or
  the Typer app composition site — recon below) to register the
  `tutorial` subcommand group with the `warm-cache` action.
- Create: `tests/unit/cli/test_tutorial_warm_cache.py` — CLI behavioural
  tests against a temp `data_dir`.

**Why this task exists (Systems finding R2-S6, 2026-05-19).** The
deployment runbook at §"Cache warming (post-deploy, post-restart)"
documents `elspeth tutorial warm-cache` as a **mandatory** post-deploy
step, but the command itself was never implemented anywhere in
`src/elspeth/`. Operators following the runbook would hit
`Error: No such command 'tutorial'`. The brief's no-deferrals directive
(2026-05-19) requires landing this here rather than punting to a
phantom sibling plan.

**Functional contract.** From the deployment host, against the deployed
model configuration:

```bash
elspeth tutorial warm-cache
```

Effect: fires the canonical seed prompt through the same run-path that
an interactive tutorial user would exercise, and writes the resulting
cache entry into `<data_dir>/tutorial_cache/` via the same
`TutorialCache.store(...)` code path (Task 5). On success the next
live user hits the cache and pays nothing. The CLI does NOT require a
real user account or session — it synthesises a deterministic
warm-cache session and a temporary user identifier (see Step 2
recon) — but it DOES record the warm-run in the Landscape under that
synthetic identifier so an auditor can spot warm-cache events
distinctly from live user runs.

Exit codes:

- `0` — warm-cache write succeeded; cache file now present at the
  expected path.
- `1` — recoverable failure (e.g., model API rate-limited; cache
  remains in its prior state). The CLI exits non-zero so deployment
  scripts can fail loudly rather than silently shipping with a cold
  cache.
- `2` — unrecoverable failure (config invalid, data_dir not writable,
  cache path mis-typed). Distinguished from `1` so operators can
  triage.

- [ ] **Step 1: Reconnaissance — locate the top-level CLI registration.**

```bash
grep -rn "typer.Typer\|@app.command\|@app\.add_typer" \
  src/elspeth/cli.py src/elspeth/cli/ src/elspeth/__main__.py 2>/dev/null \
  | grep -v __pycache__ | head -10
```

Identify:

1. The top-level Typer app handle (likely `app` in `src/elspeth/cli.py`).
2. Whether subcommand groups are registered via `app.add_typer(...)`
   (preferred) or via flat `@app.command(...)` decorators.
3. The existing pattern for subcommands with their own subcommand
   tree (`elspeth purge`, `elspeth validate`, etc. — find a
   precedent and copy its shape).

- [ ] **Step 2: Reconnaissance — locate the synthetic-user / synthetic-session shape.**

The warm-cache CLI invokes the run-path without an HTTP layer, so it
needs a synthetic `(user, session_id)` pair the run-path will accept.
Two options:

(a) **Deterministic synthetic identifiers.** Construct
    `UserIdentity(user_id="elspeth-warm-cache", username="elspeth-warm-cache")`
    and a session UUID derived from the canonical prompt + model
    (deterministic so multiple warm-cache invocations don't
    accumulate Landscape rows).

(b) **Reuse an existing test-fixture helper.** Find whatever
    `tests/integration/web/conftest.py` uses to construct an
    in-memory user; lift it into a `src/elspeth/cli/_warm_cache_identity.py`
    helper that both the CLI and the tests share.

Operator decision (2026-05-19): option (a) is canonical — the
deterministic identifiers surface the warm-cache run cleanly in audit
queries (`SELECT * FROM run_attributions WHERE initiated_by_user_id =
'elspeth-warm-cache'`).

- [ ] **Step 3: Write the failing test.**

Create `tests/unit/cli/test_tutorial_warm_cache.py`:

```python
"""Tests for the `elspeth tutorial warm-cache` CLI subcommand (R2-S6)."""

from __future__ import annotations

from pathlib import Path
from typer.testing import CliRunner

from elspeth.cli import app


def test_warm_cache_succeeds_with_default_data_dir(
    tmp_path: Path, monkeypatch
) -> None:
    """Invoking `elspeth tutorial warm-cache` writes a cache entry.

    The cache file must be present at the expected path after the
    command exits 0.
    """
    monkeypatch.setenv("ELSPETH_WEB__DATA_DIR", str(tmp_path))
    runner = CliRunner()
    result = runner.invoke(app, ["tutorial", "warm-cache"])
    assert result.exit_code == 0, result.output

    cache_dir = tmp_path / "tutorial_cache"
    files = list(cache_dir.glob("*.json"))
    assert len(files) == 1, f"expected one cache file, got {files}"


def test_warm_cache_writes_parseable_entry(
    tmp_path: Path, monkeypatch
) -> None:
    """The cache file is parseable as a TutorialCacheEntry."""
    from elspeth.web.preferences.tutorial_cache import (
        CANONICAL_SEED_PROMPT,
        TutorialCache,
        TutorialCacheEntry,
    )

    monkeypatch.setenv("ELSPETH_WEB__DATA_DIR", str(tmp_path))
    runner = CliRunner()
    runner.invoke(app, ["tutorial", "warm-cache"])

    cache = TutorialCache(cache_dir=tmp_path / "tutorial_cache")
    # The CLI uses the deployment's configured composer_model; the
    # test reads back via the same lookup contract (canonical prompt +
    # whatever model the CLI used). The model resolution path is the
    # subject of Step 4's recon — bind the assertion to the resolved
    # model id once Step 4 lands.
    entry = cache.lookup(
        CANONICAL_SEED_PROMPT,
        model_id=_resolve_test_model_id(),  # Step 4 helper
    )
    assert entry is not None
    assert isinstance(entry, TutorialCacheEntry)
    assert entry.canonical_prompt == CANONICAL_SEED_PROMPT
    assert entry.rows  # non-empty — the warm run produced output


def test_warm_cache_reports_path_and_size_on_success(
    tmp_path: Path, monkeypatch
) -> None:
    """Stdout includes the cache file path and byte size after a successful write."""
    monkeypatch.setenv("ELSPETH_WEB__DATA_DIR", str(tmp_path))
    runner = CliRunner()
    result = runner.invoke(app, ["tutorial", "warm-cache"])
    assert result.exit_code == 0
    assert str(tmp_path / "tutorial_cache") in result.output
    assert "bytes" in result.output.lower() or "size" in result.output.lower()


def test_warm_cache_exits_nonzero_on_data_dir_unwritable(
    tmp_path: Path, monkeypatch
) -> None:
    """If the data_dir is unwritable, exit code is 2 (unrecoverable)."""
    monkeypatch.setenv("ELSPETH_WEB__DATA_DIR", "/no/such/path/nowhere")
    runner = CliRunner()
    result = runner.invoke(app, ["tutorial", "warm-cache"])
    assert result.exit_code == 2
```

Expected: FAIL — `tutorial` subcommand does not yet exist on the CLI.

- [ ] **Step 4: Implement the CLI subcommand.**

Create `src/elspeth/cli/tutorial.py`:

```python
"""`elspeth tutorial` CLI subcommand group.

Phase 4A.9: implements the `warm-cache` action documented in
21a1 §"Cache warming (post-deploy, post-restart)". Mandatory on every
fresh deploy that included a sessions DB delete; the smoke task in
21b2 verifies it executed.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from uuid import NAMESPACE_URL, uuid5

import typer

from elspeth.web.auth.models import UserIdentity
from elspeth.web.config import WebSettings
from elspeth.web.preferences.tutorial_cache import (
    CANONICAL_SEED_PROMPT,
    TutorialCache,
    TutorialCacheEntry,
    _compute_key,
)

tutorial_app = typer.Typer(help="Tutorial cache management.")


@tutorial_app.command("warm-cache")
def warm_cache() -> None:
    """Fire the canonical seed prompt through the run-path and write the
    resulting cache entry. Mandatory after every fresh deploy."""
    try:
        settings = WebSettings()
    except Exception as exc:  # config invalid — exit 2 (unrecoverable)
        typer.echo(f"warm-cache: invalid WebSettings: {exc}", err=True)
        raise typer.Exit(code=2) from exc

    cache_dir = settings.tutorial_cache_dir
    if cache_dir is None:
        typer.echo(
            "warm-cache: WebSettings.tutorial_cache_dir is None — "
            "Task 2's model_validator did not populate the default. "
            "Check src/elspeth/web/config.py.",
            err=True,
        )
        raise typer.Exit(code=2)
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        typer.echo(f"warm-cache: cannot create cache dir: {exc}", err=True)
        raise typer.Exit(code=2) from exc

    # Deterministic synthetic identifiers (Step 2 option a).
    synthetic_user = UserIdentity(
        user_id="elspeth-warm-cache",
        username="elspeth-warm-cache",
    )
    synthetic_session_id = str(
        uuid5(NAMESPACE_URL, f"warm-cache:{CANONICAL_SEED_PROMPT}:{settings.composer_model}")
    )

    # Drive the same canonical-pipeline run path Task 7 uses, but
    # without the HTTP layer. The exact helper depends on Step 1/2
    # recon — likely a shared `run_canonical_tutorial_pipeline`
    # function extracted alongside Task 7.1 so both the HTTP route
    # handler and this CLI invoke the same code path.
    from elspeth.web.composer.tutorial_service import (
        run_canonical_pipeline_without_request,  # Task 7.1 recon target
    )

    try:
        result = asyncio.run(
            run_canonical_pipeline_without_request(
                user=synthetic_user,
                session_id=synthetic_session_id,
                settings=settings,
            )
        )
    except Exception as exc:
        # Recoverable: rate-limit, transient model API error, etc.
        # The cache remains in its prior state.
        typer.echo(f"warm-cache: run failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    # Resolve the model id the run actually used (compound key —
    # composer_model:transform_model — same shape as
    # `_model_id_for_pipeline` in Task 7).
    cache = TutorialCache(cache_dir=cache_dir)
    entry = TutorialCacheEntry(
        canonical_prompt=CANONICAL_SEED_PROMPT,
        model_id=result["model_id"],
        cached_at=datetime.now(UTC),
        rows=result["rows"],
        source_data_hash=result["source_data_hash"],
        llm_call_count=result["llm_call_count"],
        pipeline_yaml=result["pipeline_yaml"],
    )
    cache.store(entry)

    # Report path + size on stdout so deployment scripts can grep.
    key = _compute_key(entry.canonical_prompt, entry.model_id)
    written = cache_dir / f"{key}.json"
    typer.echo(
        f"warm-cache: OK\n"
        f"  cache_dir = {cache_dir}\n"
        f"  cache_key = {key}\n"
        f"  size      = {written.stat().st_size} bytes"
    )
```

Register the subcommand group in the top-level CLI module (Step 1
recon identified the exact site):

```python
# src/elspeth/cli.py (or wherever the top-level Typer app lives)
from elspeth.cli.tutorial import tutorial_app

app.add_typer(tutorial_app, name="tutorial")
```

- [ ] **Step 5: Run test to verify it passes.**

```bash
.venv/bin/python -m pytest tests/unit/cli/test_tutorial_warm_cache.py -v
```

Expected: PASS — all CLI tests green.

- [ ] **Step 6: Verify the deployment runbook reference points at this command.**

Confirm §"Cache warming (post-deploy, post-restart)" in this plan
references `elspeth tutorial warm-cache` (it should — that's where
this task was extracted from). No new doc edit required; this is a
consistency check.

- [ ] **Step 7: Commit.**

```bash
git add src/elspeth/cli/tutorial.py \
        src/elspeth/cli.py \
        tests/unit/cli/test_tutorial_warm_cache.py
git commit -m "feat(cli): elspeth tutorial warm-cache subcommand (Phase 4A.9)"
```

---

## What Part 1 leaves the backend in

After Tasks 0–9 land:

- `user_preferences_table` carries `tutorial_completed_at: datetime | None`, with `SESSION_SCHEMA_EPOCH` bumped and the Tier-1 read guard in `_row_to_prefs` rejecting non-`datetime` values.
- `ComposerPreferences` and `UpdateComposerPreferencesRequest` expose the field with three-state PATCH semantics (absent / datetime / `null`) honoured end-to-end via Pydantic v2 `model_fields_set`.
- `PreferencesService` write path returns the extended tuple shape `(insert_mode, resolved_banner, resolved_tutorial, tutorial_changed, wrote)`; the `composer.preferences.patch_total` counter carries the new `tutorial_changed` attribute.
- `runs_table` carries three new audit-story columns (`llm_call_count` nullable, `seeded_from_cache` NOT NULL with `server_default=0`, `cache_key` nullable) per R2-S4 (2026-05-19) — `source_data_hash` and `plugin_versions` remain at row/node level and Task 7.2 aggregates them at query time. The existing `started_at` is reused unchanged.
- The `elspeth tutorial warm-cache` CLI subcommand (Task 9) ships, so the deployment runbook's mandatory post-deploy cache-warming step actually has a command to invoke.
- `LandscapeWriteRepository` and the per-request `Depends`-graph providers exist and are wired into the FastAPI app composition site (no `app.state` shim).
- `src/elspeth/web/preferences/tutorial_cache.py` ships with SHA-256 keying on `(canonical_prompt, model_id)`, atomic temp-file-and-rename writes, Tier-1 corruption crash, and the operational guarantees documented in Task 5.
- The composer run-path consults the cache under tutorial mode, populates it on the canonical-seed run when every row succeeded, and bypasses cache for users in `freeform` mode or with `tutorial_completed_at IS NOT NULL`.
- `_replay_cached_content_to_landscape` is implemented and synthesises a real, owned-by-the-current-session Landscape entry on cache hit, with `seeded_from_cache: true` and the cache key recorded.

Not yet shipped in Part 1, continued in [21a2-phase-4-backend-part-2.md](21a2-phase-4-backend-part-2.md):

- The tutorial-specific endpoint `POST /api/tutorial/run` and its service (Task 7.1).
- The audit-story endpoint `GET /api/sessions/{session_id}/runs/{run_id}/audit-story` and its service, reading entirely from real Landscape rows (Task 7.2).
- The frontend API client functions `runTutorialPipeline`, `getRunAuditSummary`, and the `renameSession` rename, plus their unit tests (Task 7.3).
- The launch-critical tutorial telemetry counters (`composer.tutorial.complete_total`, `composer.tutorial.skip_total`, `composer.tutorial.abandon_total`) and their emit sites (Task 8).

Continue in [21a2-phase-4-backend-part-2.md](21a2-phase-4-backend-part-2.md).
