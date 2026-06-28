# Phase 4 — First-Run Hello-World Tutorial

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement these plans task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Status:** 2026-05-15. Plan-of-record for Phase 4 of the composer UX redesign.

**Goal:** Ship the first-run hello-world tutorial described in
[04-first-run-tutorial.md](04-first-run-tutorial.md): a forced sequential 6-turn
flow that runs on a user's very first composer session, builds a real pipeline
from the canonical seed prompt ("create a list of 5 government web pages and use
an LLM to rate how cool they are"), runs it, exposes the audit trail against the
user's own pipeline, and resolves the default-mode preference. The tutorial
pipeline is preserved as a real session in the user's history — not a throwaway.

**Architecture:** Schema-then-service-then-route-then-store-then-widgets. The
tutorial is **frontend-orchestrated** over backend primitives that already exist
(after Phases 1A/B, 5a, 5b ship). Phase 4 adds:

- One column on `user_preferences_table` (`tutorial_completed_at`) — Tier-1 user
  preference state.
- A flat-file `tutorial_cache` keyed by `(canonical_prompt_sha256, model_id)` to
  cache the canonical-seed-prompt run output and keep the tutorial fast and
  cheap. Cache hits cause the run-path to synthesise a new current-session-owned
  Landscape entry from the cached deterministic content (`rows`,
  `source_data_hash`, `pipeline_yaml`); the entry is marked
  `seeded_from_cache: true` and carries the cache key for cross-run lineage
  joins — see §Auditability boundary.
- A `HelloWorldTutorial.tsx` container plus six turn components.
- Frontend routing logic: on session bootstrap, if the user has no
  `tutorial_completed_at`, the composer renders the tutorial container in place
  of the normal composer surface.
- Finalisation: at turn 6 the tutorial writes `tutorial_completed_at` AND
  `default_mode` to `user_preferences` and preserves the produced pipeline as a
  named session.

**Tech Stack:** SQLAlchemy Core, FastAPI, Pydantic v2, pytest, React 18,
TypeScript, Zustand, Vitest, Playwright.

**Sibling plans:**
- This document (21) — overview, scope, dependencies, trust-tier check, file
  inventory, and risks.
- [21a1-phase-4-backend-part-1.md](21a1-phase-4-backend-part-1.md) — backend infrastructure: schema column, service
  extension, route extension, tutorial cache, run-path integration.
- [21a2-phase-4-backend-part-2.md](21a2-phase-4-backend-part-2.md) — backend endpoints + telemetry: tutorial-run endpoint, audit-story endpoint, frontend API client, launch telemetry counters.
- [21b1-phase-4-frontend-part-1.md](21b1-phase-4-frontend-part-1.md) — store,
  state machine, copy module, turns 1/2/2b/3 (frontend Tasks 1–7).
- [21b2-phase-4-frontend-part-2.md](21b2-phase-4-frontend-part-2.md) — turns
  4/5/6, container, App.tsx detection, frontend API client (`runTutorialPipeline`,
  `getRunAuditSummary`, `renameSession`, `deleteTutorialOrphans`), orphan-session cleanup wire-up,
  Reset-tutorial link in `ComposerPreferencesPanel`, Vitest integration,
  Playwright E2E, staging smoke (frontend Tasks 7.5–16).

**Roadmap reference:** [00-implementation-roadmap.md](00-implementation-roadmap.md).
Design spec: [04-first-run-tutorial.md](04-first-run-tutorial.md).

---

## Scope boundaries

**In scope (this phase):**

- Extend `user_preferences_table` with `tutorial_completed_at: timestamp NULL`.
- Extend `PreferencesService` and the GET/PATCH route to expose
  `tutorial_completed_at`.
- Add the tutorial-run cache (flat file, JSON, SHA-256-keyed).
- Wire the cache into the composer run path under "tutorial mode."
- Extend the frontend `preferencesStore` to read `tutorialCompleted` from the
  same payload (consistent with how `defaultMode` and `bannerDismissedAt` are
  already wired by Phase 1B).
- App-bootstrap detection of "first session" state per user; route to tutorial
  container.
- `HelloWorldTutorial.tsx` 6-turn state machine.
- Six turn components: `TutorialTurn1Welcome.tsx`,
  `TutorialTurn2Describe.tsx`, `TutorialTurn2bShowBuilt.tsx`,
  `TutorialTurn3Graph.tsx`, `TutorialTurn4Run.tsx`,
  `TutorialTurn5AuditStory.tsx`, `TutorialTurn6ModeChoice.tsx`.
- Skip affordance in turn 1 (fast-forwards to turn 6 only).
- Finalisation: write `tutorial_completed_at` AND `default_mode`; preserve the
  produced pipeline as a named session ("hello-world (cool government pages)").
- Vitest unit tests for the turn components, store extension, and routing.
- Playwright E2E test of the full new-user journey.
- Staging smoke deploy at `elspeth.foundryside.dev`.

**Out of scope (handled elsewhere or post-launch):**

- Re-take tutorial from settings (Open Question C3 — Phase 8 Task 6 ships
  this affordance). The Phase 4 PATCH contract permits nullification via
  `{"tutorial_completed_at": null}` so Phase 8 can clear the column with no
  additional Phase 4 changes; see §"Cross-plan contract" in
  [21a1-phase-4-backend-part-1.md](21a1-phase-4-backend-part-1.md).
- A "rename" transform alternative to the LLM-rate transform for the canonical
  seed (Open Question C1; recommended call is `(a) LLM with aggressive cache`,
  which is what this plan implements).
- File upload of any kind, plugin catalog, multi-step wizards, gates/forks,
  YAML view — per design doc 04 §"What the tutorial deliberately avoids."
- Org-default override of the tutorial gate (out of scope per Q F2 in roadmap).
- A `tutorial_runs_table` or per-attempt persistence — the cache is for output
  reuse, not for tracking who's taken the tutorial (that's
  `tutorial_completed_at`).
- Localisation of tutorial copy. English only for now; copy is hard-coded.

## Trust tier check (per CLAUDE.md)

The tutorial flow has three distinct trust surfaces, each with a specific tier:

1. **User-edited seed prompt** (turn 2 input box): **Tier 3** — external. The
   user can submit anything, including the canonical seed unchanged. The prompt
   is forwarded to the existing composer LLM-driven `set_pipeline` path, which
   already has Tier-3 boundary handling. The tutorial does **not** add a new
   Tier-3 boundary; it reuses the existing one. If the user's edit produces a
   validation error, the tutorial surfaces it inline ("here's what couldn't be
   built and why") with a "restore canonical seed" affordance — per design
   doc 04 §Risks.

2. **`tutorial_completed_at` (read)**: **Tier 1** — our data. Read from
   `user_preferences_table`. The Phase 1A read-side guard already crashes on a
   corrupt `default_composer_mode` value; this plan extends the same pattern to
   `tutorial_completed_at`. A non-NULL value must parse as a `datetime`; if it
   doesn't, crash with `RuntimeError` naming the user and the offending value.
   The frontend treats any non-NULL value (regardless of when) as "tutorial
   done."

3. **Tutorial cache file** (`~/.elspeth_web/tutorial_cache/<sha>.json` or
   equivalent — operator-configurable): **Server-generated content cache**.
   Operationally follows Tier-1 rules (crash on corruption, miss on absence,
   no live-LLM fallback on corruption); conceptually the data is LLM-derived,
   not Tier-1 "our data" in the CLAUDE.md sense. We wrote the file;
   corruption is a fault we must surface, not paper over. If the file exists,
   it must JSON-parse to the expected shape; if not, crash. The cache is an
   **optimisation, not a fallback**. Cache misses (key not present) are not
   faults — they fall through to a real LLM call. See
   `21a1-phase-4-backend-part-1.md` Task 5 §"Tier classification" for the full
   framing.

4. **LLM-generated source URLs** (the 5 government URLs returned by the LLM in
   turn 2's `set_pipeline` call): **Tier 3 at the composer LLM boundary** —
   the existing composer LLM-call wrappers handle this. The web_scrape
   transform then receives the URLs as Tier-2 pipeline data (validated by the
   source). The tutorial layer does no additional handling; this is purely
   re-using existing infrastructure.

**Why this matters for turn 5 ("the load-bearing turn"):** the design doc 04
promises real audit hashes ("Hash a7f3e2… of your URLs"), real prompt-template
recordings (from the Phase 5b `interpretation_events_table`), and real run
hashes. None of these are canned. The cache, when it hits, returns a previously
recorded run output that contains real-but-pre-computed hashes; on a cache
miss, the live run produces fresh hashes. Both paths deliver on the design
doc's "must deliver" promises in section "What the tutorial promises and must
deliver."

## Dependency check — all dependencies must be shipped before this phase starts

This phase has **hard dependencies** on three earlier phases. Each is verified
during Task 0 (preflight) of plan 21a. If any verification fails, Phase 4
implementation halts and the operator is informed.

| Dependency | What it provides | Verification |
|---|---|---|
| **Phase 1A** (backend `user_preferences_table`) | The table to extend with `tutorial_completed_at`. The schema-tests pattern to extend. The `PreferencesService` to extend. The GET/PATCH route to extend. | Read `src/elspeth/web/sessions/models.py`; confirm `user_preferences_table` exists with columns `user_id`, `default_composer_mode`, `banner_dismissed_at`, `updated_at` and no others. |
| **Phase 1B** (frontend `preferencesStore`) | The store to extend with `tutorialCompleted`. The PATCH wiring pattern. | Read `src/elspeth/web/frontend/src/stores/preferencesStore.ts`; confirm it exists, exposes `defaultMode` and `bannerDismissedAt`, and has a `setBannerDismissed`-shaped PATCH helper. |
| **Phase 5a** (dynamic-source-from-chat) | The `inline_blob` source path used by turn 2. The `set_pipeline` LLM tool-call that produces the source from the user's one-sentence description. | Read `src/elspeth/web/composer/skill.py` (or wherever the composer skill prompt lives — confirm during recon); confirm the LLM is biased toward `inline_blob` for short user inputs. |
| **Phase 5b** (surface-the-LLM's-interpretation) | The `interpretation_events_table` row that turn 2b's interpretation-acceptance event writes. Turn 5 cites this row by name in the audit-trail story ("Your accepted definition of 'cool' — recorded as a prompt template"). | Read `src/elspeth/web/sessions/models.py`; confirm `interpretation_events_table` exists. Read `src/elspeth/web/composer/interpretation_events/service.py` (or equivalent); confirm it exposes the `resolve_interpretation_event` / `list_interpretation_events` calls per the 5b plan. |

**Phase 7 (catalog reshape)** is deliberately **not** a dependency. The
tutorial avoids the catalog entirely — per design doc 04 §"What the tutorial
deliberately avoids": *"No vocabulary yet to use a catalog."* The catalog is
introduced in the user's first **real** session, not the tutorial.

## Open-question decisions adopted (from roadmap §A)

| # | Question | Adopted call | Implementation impact |
|---|---|---|---|
| C1 | Tutorial transform: rename, LLM, or both? | **(a) LLM with aggressive cache** | The canonical seed produces a pipeline with the existing `web_scrape` + `llm_rate` transforms. No rename-transform alternative is shipped. The cache is the cost-control mechanism. |
| C2 | Tutorial-skip affordance | **(b) subtle skip link in turn 1 → fast-forwards to turn 6 only** | Turn 1 component includes a small "I've used ELSPETH before, skip this" link. Clicking it sets `tutorial_completed_at` to now AND advances directly to turn 6's mode-choice — without build/run. The vocabulary teaching is lost (acceptable for returning users per design doc 04 §Risks). |
| C3 | Re-take tutorial from settings | **In Phase 4** (Reset link in `ComposerPreferencesPanel.tsx` clears `tutorial_completed_at`) | Phase 4 ships an in-settings "Reset tutorial" link added to the existing `ComposerPreferencesPanel.tsx` (21b2 §"Task 14.5"); the link PATCHes `{"tutorial_completed_at": null}` per the cross-plan contract (see §"Cross-plan contract — `tutorial_completed_at` PATCH semantics" in [21a1-phase-4-backend-part-1.md](21a1-phase-4-backend-part-1.md)). This is the always-available escape hatch for a user who skipped or completed and wants to retake without operator intervention. Phase 8 Task 6's polished "Replay hello-world tutorial" button is a separate, foregrounded affordance built on the same PATCH contract; per project memory `feedback_no_calendar_shipping_commitments`, Phase 8 may slip indefinitely, so the Phase 4 link ensures retake is reachable from launch. No sibling-plan deferral: the UI lands in `ComposerPreferencesPanel.tsx` inside this phase. |

## "Shifting the Burden" risk explicitly acknowledged

Per the Phase 1A review (systems reviewer's finding), each subsequent
column-addition to `user_preferences_table` wipes prior user state because the
operator deletes the DB on schema change. This is the **second** column-add
since Phase 1A:

- Phase 1A: created the table.
- Phase 1A→1B handover: first DB delete.
- **Phase 4 (this plan): adds `tutorial_completed_at` — second DB delete.**
- Phase 8 (telemetry): may add a third column — third DB delete.

This is "Shifting the Burden" — every column-add defers the structural fix
(Alembic or equivalent migration runner). This plan **does not** introduce
Alembic; that decision is owned by the roadmap. This plan **does** add the
column and document the DB-delete operator action explicitly in plan 21a
Task 1's commit message and in plan 21b's smoke-deploy task.

The operator is reminded that after this phase ships, all existing user state
in `user_preferences_table` is wiped:

- `default_composer_mode` resets to the server-side `"guided"` default.
- `banner_dismissed_at` resets to NULL (banners may re-fire if Phase 1A's
  detection logic considers a NULL banner-dismissed-at as "show banner").

For test environments where operators have been clicking through the existing
Phase 1A/B flow, this is annoying but acceptable: the alternative is starting
the Alembic project, which is a much larger commitment.

## File inventory (across 21a + 21b)

### New (21a — backend)

- `src/elspeth/web/preferences/tutorial_cache.py` — flat-file cache for
  canonical-seed-prompt runs.
- `tests/unit/web/preferences/test_tutorial_cache.py` — cache tests.
- `tests/integration/web/test_tutorial_cache_run_integration.py` — verifies
  the composer run path consults the cache in tutorial mode.

### Modified (21a — backend)

- `src/elspeth/web/sessions/models.py` — add `tutorial_completed_at` column to
  `user_preferences_table`.
- `src/elspeth/web/preferences/models.py` — extend `ComposerPreferences` and
  `UpdateComposerPreferencesRequest` with `tutorial_completed_at`.
- `src/elspeth/web/preferences/service.py` — extend `_row_to_prefs`,
  `get_composer_preferences`, `update_composer_preferences` to read/write the
  new column. Extend the Tier-1 read guard.
- `src/elspeth/web/preferences/routes.py` — no structural change; the
  Pydantic-model extension picks the new field up automatically through
  `response_model`.
- `tests/unit/web/preferences/test_schema.py` — extend expected-columns set
  and add presence test for the new column.
- `tests/unit/web/preferences/test_models.py` — add cases for
  `tutorial_completed_at` field.
- `tests/unit/web/preferences/test_service.py` — add cases for the new
  read/write path and the new Tier-1 guard.
- `tests/integration/web/test_preferences_routes.py` — add cases for the new
  PATCH field.
- The composer run-path file (identified during plan 21a1 Task 7's recon) —
  add a cache-consultation hook gated on `request.app.state.preferences_service`
  reporting `tutorial_completed_at is None` for the user (i.e., tutorial mode).

### New (21b — frontend)

- `src/elspeth/web/frontend/src/components/tutorial/HelloWorldTutorial.tsx` —
  container.
- `src/elspeth/web/frontend/src/components/tutorial/TutorialTurn1Welcome.tsx`
- `src/elspeth/web/frontend/src/components/tutorial/TutorialTurn2Describe.tsx`
- `src/elspeth/web/frontend/src/components/tutorial/TutorialTurn2bShowBuilt.tsx`
- `src/elspeth/web/frontend/src/components/tutorial/TutorialTurn3Graph.tsx`
- `src/elspeth/web/frontend/src/components/tutorial/TutorialTurn4Run.tsx`
- `src/elspeth/web/frontend/src/components/tutorial/TutorialTurn5AuditStory.tsx`
- `src/elspeth/web/frontend/src/components/tutorial/TutorialTurn6ModeChoice.tsx`
- `src/elspeth/web/frontend/src/components/tutorial/tutorialMachine.ts` —
  6-turn state machine (typed step union + reducer).
- `src/elspeth/web/frontend/src/components/tutorial/copy.ts` — extracted copy
  blocks (testability + future i18n hook).
- Test files: one `.test.tsx` per component, one for `tutorialMachine.ts`,
  one for the container's integration with the store.
- `tests/e2e/tutorial.e2e.spec.ts` (or per the project's Playwright location)
  — full new-user journey.

### Modified (21b — frontend)

- `src/elspeth/web/frontend/src/stores/preferencesStore.ts` — add
  `tutorialCompleted: boolean` (derived from `tutorialCompletedAt` non-null)
  and `markTutorialCompleted(defaultMode)` action. Phase 8 Task 6's retake
  path reuses the same PATCH API client function (with body
  `{tutorial_completed_at: null}`) — Phase 4 does **not** add a separate
  `clearTutorialCompleted` action; Phase 8 calls
  `api.updateComposerPreferences({ tutorial_completed_at: null })` and then
  re-bootstraps the store.
- `src/elspeth/web/frontend/src/api/client.ts` — extend the
  preferences-API typing to include `tutorial_completed_at`. The TypeScript
  request type must allow `null` (not just `undefined`) so the Phase 8
  retake call type-checks.
- `src/elspeth/web/frontend/src/App.tsx` — bootstrap-time check; route to
  `HelloWorldTutorial` if `preferencesStore.tutorialCompleted === false` AND
  the user has no active sessions (defensive belt-and-braces — but the
  per-user `tutorial_completed_at` is the primary gate).
- `src/elspeth/web/frontend/src/stores/sessionStore.ts` — minor extension if
  the tutorial needs to name the session "hello-world (cool government pages)"
  via an existing rename path; confirmed during plan 21b recon.
- `src/elspeth/web/frontend/src/api/client.ts` — four function additions
  (`runTutorialPipeline`, `getRunAuditSummary`, `deleteTutorialOrphans`)
  plus rename `updateSessionTitle` → `renameSession` (single rename, all
  call sites updated atomically; No Legacy Code Policy — no alias). Owned
  by **21b2 Task 7.5** (relocated from 21a2 Task 7.3 — these are frontend
  client symbols; they live with the frontend plan).
- `src/elspeth/web/frontend/src/components/settings/ComposerPreferencesPanel.tsx`
  — add the "Reset tutorial" link inside the `ComposerPreferencesForm`
  body. Owned by **21b2 Task 14.5** (resolves Open Question C3 in-phase;
  no sibling-plan deferral).

### Not modified

- `src/elspeth/web/auth/models.py` — `UserIdentity` / `UserProfile`
  unchanged.
- `src/elspeth/web/sessions/service.py` — session creation unchanged. The
  tutorial creates an ordinary session via the existing API and renames it.
- Phase 5a / 5b backend code — used as-is.

## Implementation order (across 21a + 21b)

**Strict order (each blocks the next):**

1. (21a) Preflight: verify Phase 1A, 1B, 5a, 5b are shipped. Abort if not.
2. (21a) Schema: extend `user_preferences_table` with `tutorial_completed_at`.
3. (21a) Pydantic model: add the field to `ComposerPreferences` and
   `UpdateComposerPreferencesRequest`.
4. (21a) Service: extend `_row_to_prefs`, `get_composer_preferences`,
   `update_composer_preferences`. Extend the Tier-1 guard.
5. (21a) Route tests: verify the GET response and PATCH body accept and round-
   trip the new field.
6. (21a) Tutorial cache: implement and unit-test the flat-file cache.
7. (21a) Run-path integration: wire cache consultation into the composer run
   path, gated on tutorial mode.
8. (21b) Store: extend `preferencesStore` to expose `tutorialCompleted` and
   `markTutorialCompleted(defaultMode)`.
9. (21b) Detection: extend `App.tsx` to route to the tutorial when
   `tutorialCompleted === false`.
10. (21b) Frontend API client surface (`runTutorialPipeline`,
    `getRunAuditSummary`, `deleteTutorialOrphans`, rename
    `updateSessionTitle` → `renameSession`). Lands before turn 4/5/6
    consumers; see 21b2 §"Task 7.5".
11. (21b) Container + state machine. Container's mount-time orphan-cleanup
    call (`DELETE /api/tutorial/orphans`) fires here — see 21b2 §"Task 11"
    Step 4.
12. (21b) Six turn components, in arc order (1 → 2 → 2b → 3 → 4 → 5 → 6).
13. (21b) Finalisation: at turn 6, write both preferences and rename the
    session.
14. (21b) Skip affordance in turn 1.
15. (21b) Vitest integration: container with mock store + mock API.
16. (21b) Reset-tutorial link in `ComposerPreferencesPanel.tsx`
    (resolves Open Question C3 in-phase — see 21b2 §"Task 14.5").
17. (21b) Playwright E2E: brand-new user journey.
18. (21b) Staging smoke at `elspeth.foundryside.dev`.

Each task is TDD-shaped (write failing test, watch it fail, implement, watch
it pass, commit). Plans 21a and 21b each contain their own per-task TDD
breakdown.

## PR strategy

Phase 4 lands as **two PRs against `RC5.2`**, mirroring the 21a / 21b split.
Architecture review (A8) flagged this as a 30-file change with a weighted
blast-radius score of 27 ("Very High"); shipping it as a single PR would
make rollback an all-or-nothing operation across two distinct system layers.

- **PR-21a** — backend half. Tasks 0 through 9 split across
  `21a1-phase-4-backend-part-1.md` (Tasks 0–7.0, 7, **Task 9 `elspeth
  tutorial warm-cache` CLI subcommand added per R2-S6, 2026-05-19** —
  preflight, schema column, Pydantic model, service, route tests,
  tutorial cache, run-path integration, Landscape audit-story schema,
  warm-cache CLI) and `21a2-phase-4-backend-part-2.md` (Tasks 7.1, 7.2,
  7.4 orphan-cleanup endpoint, 8 — tutorial-run endpoint, audit-story
  endpoint, `DELETE /api/tutorial/orphans` endpoint, **and the launch
  telemetry counters added by P14: `composer.tutorial.complete_total`,
  `composer.tutorial.skip_total`, `composer.tutorial.abandon_total`**).
  The frontend API client surface (originally drafted as 21a2 Task 7.3)
  is relocated to the frontend plan per M-R2-2 (2026-05-19) — see 21b2
  §"Task 7.5: Frontend API client surface — `runTutorialPipeline`,
  `getRunAuditSummary`, `renameSession`, `deleteTutorialOrphans`" for the
  relocated work. The two halves are co-dependent and land in a single
  PR. **Must merge first.**
- **PR-21b** — frontend half. Tasks 1 through 16 across
  `21b1-phase-4-frontend-part-1.md` (Tasks 1–7) and
  `21b2-phase-4-frontend-part-2.md` (Tasks 7.5, 8–10, 11 with orphan-cleanup
  wire-up, 12–14, 14.5 Reset-link in `ComposerPreferencesPanel`, 15
  Playwright, 16 staging smoke). **Depends on PR-21a; do not merge to
  `RC5.2` before PR-21a is on `RC5.2`.**

Rationale: blast-radius isolation (a backend regression and a frontend
regression are independently revertable), and the frontend's Vitest +
Playwright tests cannot run green until PR-21a's `/api/tutorial/run` and
`/api/sessions/.../audit-story` endpoints exist on the target branch.

## Auditability boundary (CLAUDE.md attributability test)

The tutorial **is** a real session: the pipeline produced during the tutorial
is a real pipeline, its run is a real run, and every artifact recorded by the
Landscape audit trail is recorded for that run exactly as it would be for a
non-tutorial session.

What the tutorial layer adds that is **not** in the Landscape:

- `tutorial_completed_at` — user-preference state, not pipeline state. Per the
  Phase 1A "Auditability boundary" note, composer mode and tutorial state are
  authoring-time UI affordances, not pipeline behaviour. Runtime pipeline
  execution and outputs do not vary by whether the user is in the tutorial.
- Cache hits — when the cache hits, the run-path does **not** return a foreign
  run's audit-trail snippets. Instead:
  - A new `run_id` is created under the current session via
    `_replay_cached_content_to_landscape` (21a1 Task 7). The synthesised
    Landscape entry is owned by the current session, populated from the
    cached deterministic content (`rows`, `source_data_hash`, `pipeline_yaml`).
  - The synthesised entry carries `seeded_from_cache: true` and the cache key
    (`SHA-256(canonical_prompt + ":" + model_id)`) on its metadata, which is
    how a future query joins back to the original seeding run for cross-run
    lineage.
  - `source_data_hash` carries genuine determinism evidence from the cache —
    it is the *content* hash, reproducible across runs — not a copied
    identity. Identity (`run_id`, `interpretation_event_id`) is **never**
    stored in the cache; the cache holds content, not pointers.
  - The attributability test still holds: `explain(recorder, run_id,
    token_id)` on the cache-replay run returns the cache-replay's full
    lineage, including the cache-key join to the original seeding run. No
    cross-ownership query is required.

The `attributability test`: for any output of the tutorial run, `explain(
recorder, run_id, token_id)` proves complete lineage back to source. The
tutorial does **not** weaken this — it simply uses the standard composer
plumbing to produce a real run.

## Performance budget

| Surface | Budget | Rationale |
|---|---|---|
| Cache hit lookup | < 50ms | SHA-256 of a short constant string + JSON file read of a few KB. |
| Cache miss → live run | "real LLM cost" — up to ~30 seconds for 5x LLM calls + 5x web_scrape | Per design doc 04 §Risks. Mitigated by cache hit being the dominant case. |
| Turn-to-turn transition | < 100ms | Pure React state machine; no API calls between turns 1, 2b, 3, 5, 6. Turn 2 → 2b makes one API call (compose); Turn 3 → 4 makes one API call (run); Turn 6 → done makes one API call (PATCH preferences). |
| Full tutorial (cache hit) | < 60 seconds for a quick reader | Per design doc 04 §"Turn-by-turn arc": "Target: ~3 minutes for a quick reader; ~5 minutes for someone who explores." 60 seconds is the floor (rapid click-through). |

The cache key intentionally excludes the user id so that all users share the
same cached output for the canonical seed prompt. This is correct: the
canonical run is deterministic-enough that sharing the output is the desired
behaviour (and the failure mode of not sharing — every new user pays for a
fresh LLM run — is the cost we're trying to avoid).

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| **LLM picks URLs that don't load.** | The `web_scrape` transform's error handling is already audit-aware. The tutorial's turn 4 result table shows the failure inline ("page 3: HTTP 503 — recorded in audit") and treats it as a teaching moment about robust pipelines (per design doc 04 §Risks). If the cache is in use, the cached output is selected from a known-good run with all URLs reachable at cache-creation time. |
| **Cache hit obscures provenance of LLM responses.** | On a cache hit the synthesised current-session run records `seeded_from_cache: true` and the SHA-256 cache key (`canonical_prompt:model_id`). An auditor querying `explain(recorder, run_id, token_id)` for a cache-hit run sees both the local lineage and the cache-key join back to the original seeding run. Turn 5's narration also acknowledges the cache replay in user-facing copy — neither the audit trail nor the UX hides the replay. |
| **Tutorial cache key collision.** | The key is `SHA-256(canonical_prompt + ":" + model_id)`. Collisions are cryptographically negligible. The risk is **not** collision; the risk is **stale entries**: if the model rotates (e.g., the deployment switches from claude-opus-4.7 to a newer model), the previously cached entry is for a different model and must miss. The key includes `model_id` precisely to force a miss on model rotation. There is no LRU eviction (the cache is small — one canonical key per model the deployment has used); operators may delete the cache directory at will. |
| **User edits the seed prompt to something the composer can't handle.** | Composer validation falls back to the existing error path. The tutorial surfaces the validation error in audit-readiness style ("here's what couldn't be built and why") and offers a "restore canonical seed" affordance per design doc 04 §Risks. The edited prompt is **not** cached (the cache key matches the canonical seed prompt exactly). |
| **User refreshes the page mid-tutorial.** | Refresh during the tutorial → state machine restarts at turn 1. The state machine is in-memory (Zustand-free; lives in `HelloWorldTutorial.tsx` `useReducer` state); there is **no** `sessionStorage` scaffolding. Acceptable simplification (plan-fix P10, 2026-05-19): the tutorial is ~5 minutes; canonical-seed produces a fresh pipeline each time; cache makes that fast. **No `sessionStorage` scaffolding** — a flat-key approach (`elspeth_tutorial_progress`) isn't user-scoped, so it risks cross-user contamination on shared workstations (systems S6); per-user keying would add Vitest+Playwright surface area that doesn't materially improve UX. Verified by a Playwright case in 21b2 Task 14 (`refresh mid-tutorial restarts at turn 1`) that walks to turn 3, reloads, and asserts turn 1 plus a null `sessionStorage` key. |
| **Cache corruption.** | Cache files are a server-generated content cache — operationally Tier-1 (crash on corruption, miss on absence, no live-LLM fallback) but conceptually LLM-derived. The tutorial cache reader uses Pydantic to parse the JSON file. If the parse fails, crash with the file path and the parse error chained via `from`. The operator's recovery is to delete the cache directory. **Do not** silently fall back to a live LLM call: that would mask the corruption and ship demonstrably wrong audit snippets to the user. Full framing: `21a1-phase-4-backend-part-1.md` Task 5 §"Tier classification". |
| **Concurrent first-session for the same user (two browser tabs).** | Two tabs may both decide "tutorial not yet done" and both render the tutorial container. The user finishes in tab A; PATCH writes `tutorial_completed_at`. Tab B continues rendering its tutorial container until its next preference read. Outcome: tab B may produce a second pipeline/session named "hello-world (cool government pages)". This is acceptable — the user has two pipelines they can run/edit/delete independently. No locking is added (the Phase 1A "Concurrent PATCH race window" mitigation applies: SQLite `ON CONFLICT DO UPDATE` resolves to the last write). |
| **User explicitly clicks "skip" but immediately wants to come back.** | Per the P20 resolution, Phase 4 ships an in-settings "Reset tutorial" link added to the existing `ComposerPreferencesPanel.tsx` (21b2 §"Task 14.5") that PATCHes `{"tutorial_completed_at": null}` per the cross-plan contract (see §"Cross-plan contract — `tutorial_completed_at` PATCH semantics" in [21a1-phase-4-backend-part-1.md](21a1-phase-4-backend-part-1.md)). A user who skipped or completed and wants to retake can use the link without operator intervention. The Phase 8 Task 6 "Replay hello-world tutorial" button is a separate, polished affordance built on the same PATCH contract; the Phase 4 link is the always-available escape hatch from launch onward, in case Phase 8 slips. |
| **User edits the LLM's interpretation in turn 2b but then changes their mind.** | The Phase 5b interpretation-events table records every resolution; if the user wants to revisit, the standard composer flow surfaces pending interpretation events. The tutorial's turn 2b is just a styled wrapper around the existing `InterpretationReviewTurn` from Phase 5b. The user cannot "undo" an interpretation acceptance from inside the tutorial; the design doc 04 implicit choice is that this is acceptable for a 3-minute hello-world. |
| **Cache fills with one entry per model rotation, growing forever.** | One entry per `(canonical_prompt, model)` is small (model count is small; canonical prompts is one). Filesystem footprint is bounded. No eviction policy needed. |
| **Phase 5b's `interpretation_events_table` doesn't yet record the prompt-template string.** | This is a Phase 5b dependency. The Phase 4 preflight (plan 21a Task 0) verifies it. If 5b's table doesn't yet record the prompt template, the tutorial cannot deliver on the design doc 04 promise "Your accepted definition of 'cool' — recorded as a prompt template" — in which case Phase 4 implementation halts and the operator is informed. Per CLAUDE.md "no defensive programming": the tutorial does **not** fall back to canned text. |
| **The tutorial's commitment to a 6-turn sequence might frustrate the user.** | The skip affordance in turn 1 is the operator-approved escape valve (Open Question C2 recommended call). |

## Promises the tutorial makes and the implementation requirements they impose

From design doc 04 §"What the tutorial promises and must deliver" — each
promise mapped to its Phase 4 implementation:

| Promise | Implementation |
|---|---|
| "I'll wire it up from your one sentence." | Turn 2 forwards the (canonical or user-edited) prompt to the existing composer LLM-driven `set_pipeline` path. The LLM's `inline_blob`-bias prompt from Phase 5a does the rest. |
| "Here's what I drafted." | Turn 2b reads the composition state via the standard composer state-read API (`get_pipeline_state` or equivalent) and renders source URLs + transform names + sink description. Editable elements are forwarded to existing composer edit affordances. |
| "Your accepted definition of 'cool' — recorded as a prompt template." | Turn 2b embeds the Phase 5b `InterpretationReviewTurn` for the "cool" interpretation event. The user's acceptance (or edit + acceptance) writes to `interpretation_events_table` exactly as Phase 5b specifies. The audit trail therefore has a distinct event row, not a silently baked prompt. |
| "Hash a7f3e2… of your URLs." | Turn 5 reads the `source_data_hash` (or equivalent column name — confirmed during plan 21b recon) from the **just-completed** run's Landscape entry, via the existing Landscape read API. If the cache served turn 4, turn 5 reads the cached entry's hash, which is the real hash from the cache-creation run. **Never** a canned string. |
| "If someone six months from now asks…" | The Landscape audit trail can answer; the tutorial just makes it visible. The "Explore the full audit trail" affordance opens the same audit-evidence view future sessions use (per the design doc). |

## Vocabulary

- **"Tutorial mode"** — the state where `preferences.tutorial_completed_at is
  None`. Used to gate cache-consult and the tutorial-container render.
- **"Canonical seed prompt"** — the exact string "create a list of 5
  government web pages and use an LLM to rate how cool they are". Defined as a
  module-level constant in both the backend cache key code and the frontend
  copy module.
- **"Tutorial session"** — the session created during turn 2. Named
  "hello-world (cool government pages)" at finalisation time (turn 6).
- **"Cache entry"** — a record under `~/.elspeth_web/tutorial_cache/<sha>.json`
  storing the deterministic output (`rows`, `source_data_hash`,
  `llm_call_count`, `pipeline_yaml`) of a canonical-seed-prompt run. Never
  stores identity (`run_id`, `session_id`, `interpretation_event_id`) — the
  content-not-identity invariant.
- **"Cache hit"** — `lookup(canonical_prompt, model_id)` finds an existing
  entry. The run-path synthesises a new current-session-owned Landscape entry
  from the cached content via `_replay_cached_content_to_landscape`.
- **"Cache replay"** — the synthesised Landscape entry produced on a cache
  hit. Carries `seeded_from_cache: true` plus the cache key for cross-run
  lineage joins.
- **"Cache key"** — `SHA-256(canonical_prompt + ":" + model_id)` (hex).
  Recorded on the cache-replay's metadata; appears in the audit trail and
  (short-prefix) in turn 5's narration.

## Review history

### 2026-05-19 — cross-plan contract amendment (Phase 4 ↔ Phase 8)

Pass-1 review (Systems S1) found that Phase 4's draft forbade nullification
of `tutorial_completed_at` via PATCH while Phase 8 Task 6 expected to clear
the column via the same PATCH endpoint. Resolution: Option (a) — Phase 4
permits explicit-null nullification; Phase 8 retake uses
`PATCH {"tutorial_completed_at": null}`. The same field and same column
remain a single shared contract. See 21a §"Cross-plan contract —
`tutorial_completed_at` PATCH semantics" for the canonical statement.

Edits applied in this overview:

- §Out of scope — C3 bullet updated to reference Phase 8 Task 6 and the
  shared contract (formerly "post-launch if telemetry shows demand").
- §Open-question decisions — C3 row updated to name Phase 8 Task 6 and link
  to the shared contract.
- §Risks — the "skip but wants to come back" row updated to name the
  Phase 8 retake button.
- §File inventory — `preferencesStore.ts` and `client.ts` rows note that
  Phase 8 retake reuses the existing API surface (no new
  `clearTutorialCompleted` action).

---

## Memory references

- `project_composer_first_run_tutorial` — the design call.
- `project_composer_canonical_test_case` — the seed prompt.
- `project_composer_dynamic_source_from_chat` — Phase 5a feature this depends on.
- `project_composer_default_guided_with_opt_out` — Phase 1 outcome this finalises.
- `project_db_migration_policy` — informs the "second DB-delete" operator action.
- `feedback_no_calendar_shipping_commitments` — no calendar commitments in this plan.
