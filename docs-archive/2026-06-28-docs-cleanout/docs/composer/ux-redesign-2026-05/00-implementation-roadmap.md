# 00 — Implementation Roadmap

**Status:** 2026-05-15. Companion to docs 01–11. This document does two things:

1. **Flattens** the open questions in [11-open-questions.md](11-open-questions.md) into a single
   "recommended call + reversibility" table so the operator can adjudicate by exception rather than
   walking the full list.
2. **Maps** each of the 8 implementation phases to the artifacts that exist (or don't yet) in the
   codebase, with a per-phase "ready / blocked-by" status.

It is the entry point for "I have these design docs — what do I do next?".

## TL;DR — current state (2026-05-15)

All plans are written and reviewed. Full review panel applied 2026-05-15 to Phases 2/3/4/5a/5b/6/8.
Phase 9 (migration runner) is named but not yet planned.

**Next actions:**
1. **Ship Phase 1A** — plan is REVIEWED + FIXED, no upstream blockers. Operator deletes
   sessions DB at deploy time. This is the gating action for the entire redesign.
2. **Phase 9 — plan the migration runner** — necessary BEFORE real-user deploy of any
   schema-adding phase (Phase 1A is fine for staging; Phase 1B + 2A unblocked once
   1A ships; Phase 4 and Phase 5b are blocked on Phase 9 for production deploys).
3. **Demo critical path** — minimum viable demo set: Phase 1A/1B,
   Phase 2 (2A/2B/2C), Phase 5a, **Phase 5b** (operator decision
   2026-05-18: the three-beat hello-world tutorial is demo-required;
   the third beat is the LLM-interpretation review surfaced by Phase
   5b), Phase 4 (tutorial), Phase 3A partial (session switcher only).
   The full IA cleanup (Phase 3B) and completion gestures (Phase 6)
   are demo-cosmetic; ship in parallel if time permits. Note: this
   supersedes the original systems-review framing that placed Phase 5b
   off the critical path.

Open-question adjudication is in §A. Most in-phase decisions are now resolved. B1
(audit-recorder behaviour for dynamic source) was RESOLVED 2026-05-16 with verdict (a)
"Yes, the existing inline_blob path records `source_data_hash` identically to CSV" — see
`tools.py:4414-4472`, `hashing.py:80-93`, `data_flow_repository.py:381-424` (Phase 5a plan
header). J1 (migration runner shape) is deferred to Phase 9 planning.

---

## A. Open-question adjudication table

The full discussion of each question is in [11-open-questions.md](11-open-questions.md). This table
condenses each to: **recommended call**, **reversibility**, and **what blocks if undecided**.

### Pre-Phase-0 (must decide before anything starts)

| # | Question | Recommended call | Reversibility | If undecided |
|---|---|---|---|---|
| A1 | Demo persona mix | (c) Mixed — recommendations stand | Easy | Recommendations soften silently for wrong audience |

### Pre-Phase-1 (blocks default-mode work)

| # | Question | Recommended call | Reversibility | If undecided |
|---|---|---|---|---|
| A2 | Existing-user migration | (c) Migrate-with-banner | Easy (one-time banner) | Existing users surprised on next login |
| **B4** | **(NEW) Where does `composer.default_mode` live?** | **DECIDED:** backend `user_preferences` table from the start. Schema lands in `sessions/models.py` alongside `sessions_table`; operator deletes the old DB per `project_db_migration_policy`. | Hard (one-way; once the table exists, drop the DB to revert) | n/a — decided |
| F1 | Kiosk / shared accounts | Out of scope | n/a | — |
| F2 | SSO / org-default override | Out of scope | n/a | — |

### Pre-Phase-2 (blocks audit-readiness panel)

| # | Question | Recommended call | Reversibility | If undecided |
|---|---|---|---|---|
| G1 | Trust-tier display format | (b) plain names in panel; tier numbers in Explain detail | Easy | Cosmetic only |
| G2 | Retention row when not configured | (c) trust-tier-aware | Easy | Compliance users miss the nudge |

### Pre-Phase-5a (blocks dynamic-source-from-chat)

| # | Question | Recommended call | Reversibility | If undecided |
|---|---|---|---|---|
| B1 | Does audit recorder treat dynamic-source-from-chat as a real source? | **Verify before starting Phase 5a** (run the test described in 11-§B1) | n/a — fact-finding | Tutorial turn 5 is theatre if no |

### Pre-Phase-5b (blocks interpretation surface)

| # | Question | Recommended call | Reversibility | If undecided |
|---|---|---|---|---|
| B2 | Audit-recorder supports interpretation-acceptance events? | **VERDICT (c) — new event class. RESOLVED 2026-05-15.** See [18](18-phase-5b-surface-llm-interpretation.md); DB impact in §D5. | Hard once shape ships | n/a — decided |

### Pre-Phase-6 (blocks completion gestures)

| # | Question | Recommended call | Reversibility | If undecided |
|---|---|---|---|---|
| A3 | Single- or multi-user composer? | Initial impl: shareable link → read-only inspect view | Easy (b/c upgrade path) | Save-for-review is hand-off-by-email only |
| E1 | Signed save-for-review artifact? | **RECOMMENDED (a) — HMAC-signed. Implemented in Phase 6A plan** ([19a](19a-phase-6a-backend.md)). | Hard (changes audit contract) | Compliance value reduced |

### In-Phase decisions (decide as you implement; no upstream block)

| # | Question | Recommended call | Reversibility |
|---|---|---|---|
| B3 | Audit-record YAML export? | (b) record as low-priority event | Easy |
| C1 | Tutorial transform: rename / LLM / both | **RECOMMENDED (a) — LLM with aggressive cache. Will be implemented in Phase 4** ([21](21-phase-4-hello-world-tutorial.md)). | Easy |
| C2 | Tutorial-skip affordance | **RECOMMENDED (b) — subtle skip link in turn 1.** | Easy |
| D1 | Catalog keyboard shortcut | (a) keep Ctrl+Shift+P, regroup in help | Easy |
| D2 | "Inline data from chat" — plugin or option | Likely real plugin; confirm with engine | Easy |
| D3 | Plugin "when you'd use this" prose | (a) author long-term; (c) LLM-draft to bootstrap | Easy |
| E2 | Save-for-review reviewer surface | **RECOMMENDED (a) — read-only inspect view v1. Implemented in Phase 6** ([19a](19a-phase-6a-backend.md)·[19b](19b-phase-6b-frontend.md)). | Easy |
| E3 | Run-result narrative detection | **RECOMMENDED (c) per-plugin declaration via `supports_narrative_summary` ClassVar; bootstrap (b) hardcoded list for two batch transforms. Implemented in Phase 6A** ([19a](19a-phase-6a-backend.md)). | Easy |
| H1 | Session sidebar replacement | **RECOMMENDED (c) — header dropdown + command palette. Implemented in Phase 3B** ([15b1](15b1-phase-3b-side-rail-part-1.md)). | Easy |
| H2 | Graph mini-view click target | **RECOMMENDED (a) — modal. Implemented in Phase 3B** ([15b1](15b1-phase-3b-side-rail-part-1.md)). | Easy |

### Post-launch (tune with telemetry)

| # | Question | Recommended call |
|---|---|---|
| C3 | Re-take tutorial from settings | Add if telemetry shows demand |
| I1 | Which metrics matter? | Opt-out rate + per-mode completion rate |

### Pre-Phase-9 (blocks migration runner)

| # | Question | Recommended call | Reversibility | If undecided |
|---|---|---|---|---|
| J1 | Migration runner shape: SQL DDL diff, schema migrations à la Alembic, or per-table preserve-on-recreate? | **J1 verdict: APPROVED 2026-05-16 — Option (c) per-table preserve-on-recreate.** Rationale preserved at [22-phase-9-migration-runner.md §2](22-phase-9-migration-runner.md#2-j1-adjudication-section). | Hard once shipped | All real-user deploys blocked |

---

## B. Phase-by-phase readiness summary

Plan-status legend: **REVIEWED + FIXED** — full review panel applied and changes incorporated.
**PLAN-WRITTEN** — written but not yet reviewed (review-and-fix required before implementation).
**IN PROGRESS** — plan is being drafted. See §G for the review backlog.

Implementation-readiness legend: **READY** — reviewed plan, no upstream blocker.
**BLOCKED (plan review)** — plan written but not yet reviewed.
**BLOCKED (phase X ship)** — upstream phase must ship first. **BLOCKED (plan + phase X)** — both.

| Phase | Title | Plan file(s) | Plan status | Impl. readiness |
|---|---|---|---|---|
| 1A | Backend: user_preferences table + preferences API | [12](12-phase-1a-backend.md) | REVIEWED + FIXED | READY |
| 1B | Frontend: store, opt-out surfaces, banner, smoke | [13](13-phase-1b-frontend.md) | REVIEWED + FIXED | BLOCKED (1A ship) |
| 2 | Audit-readiness panel | [14](14-phase-2-audit-readiness-panel.md) · [14a](14a-phase-2a-backend.md) · [14b](14b-phase-2b-frontend.md) · [14c](14c-phase-2c-frontend-integration.md) | REVIEWED + FIXED | BLOCKED (1B ship) |
| 3A | IA cleanup — removals | [15a1](15a1-phase-3a-removals-part-1.md) · [15a2](15a2-phase-3a-removals-part-2.md) | REVIEWED + FIXED | BLOCKED (1B ship) |
| 3B | IA cleanup — side-rail additions | [15b1](15b1-phase-3b-side-rail-part-1.md) · [15b2](15b2-phase-3b-side-rail-part-2.md) | REVIEWED + FIXED | BLOCKED (1B ship) |
| 4 | Hello-world tutorial | [21](21-phase-4-hello-world-tutorial.md) · [21a1](21a1-phase-4-backend-part-1.md) · [21a2](21a2-phase-4-backend-part-2.md) · [21b1](21b1-phase-4-frontend-part-1.md) · [21b2](21b2-phase-4-frontend-part-2.md) | REVIEWED + FIXED | BLOCKED (1/5a/5b ship) |
| 5a | Dynamic-source-from-chat | [17](17-phase-5a-dynamic-source-from-chat.md) | REVIEWED + FIXED | BLOCKED (1B ship) |
| 5b | Surface-the-LLM's-interpretation | [18](18-phase-5b-surface-llm-interpretation.md) · [18a](18a-phase-5b-backend.md) · [18b](18b-phase-5b-frontend.md) | REVIEWED + FIXED | BLOCKED (5a ship) |
| 6 | Completion gestures | [19a](19a-phase-6a-backend.md) · [19b](19b-phase-6b-frontend.md) | REVIEWED + FIXED | BLOCKED (3/5b ship) |
| 7 | Catalog reshape | [16](16-phase-7-catalog-reshape.md) · [16a](16a-phase-7a-backend.md) · [16b](16b-phase-7b-frontend.md) · [16c](16c-phase-7c-frontend-integration.md) | REVIEWED + FIXED | BLOCKED (1B ship) |
| 8 | Polish + telemetry | [20](20-phase-8-polish-and-telemetry.md) | REVIEWED + FIXED | BLOCKED (all prior ship) |
| 9 | Migration runner + caretaker-logic activation | [22](22-phase-9-migration-runner.md) | PLAN READY (J1 APPROVED 2026-05-16; full review panel applied) | BLOCKED (Phase 1A + Phase 5b ship; operator decisions #2 rollback strategy + #3 fixture-generation approach; real users) |

---

## C. Recommended ship sequencing

All plans are now written (Phase 4 in progress). The sequence below shows the **ship order** for
implementation — not planning. Phases on the same indentation level can ship in parallel.

> **Critical-path note (operator decision 2026-05-18):** Phase 5b is on the
> demo critical path. The hello-world tutorial (Phase 4) is delivered as a
> three-beat experience, and the third beat is the LLM-interpretation review
> surfaced by Phase 5b. The dependency edge `5b → 4` is therefore load-bearing
> for the demo: if 5b slips, 4 cannot ship in its demo-required form. This
> supersedes the original systems-review framing that placed Phase 5b "off the
> critical path." Within 5b, Task 0 (empirical placeholder-convention gate;
> see [18a §Task 0](18a-phase-5b-backend.md)) is the implementer's first
> action — if it fails, 5b pivots to a runtime-resolve architecture and the
> downstream timeline must be re-evaluated.

```text
1A (backend)
  └─ 1B (frontend)
       ├─ 2A → 2B → 2C   ─┐  parallel after 1B
       ├─ 3A1 → 3A2        │  (3A and 3B can run parallel with each other and with 2*)
       └─ 3B1 → 3B2   ────┘
            │
            └─ 5a (dynamic source)
                  └─ 5b-Task-0 → 5b-backend → 5b-frontend   ← critical path
                        │
                        ├─ 4 (hello-world tutorial)   ← needs 1 + 5a + 5b (5b required for the third beat)
                        └─ 6A → 6B                    ← needs 3 + 5b
7A → 7B → 7C   (independent track; can start any time after 1B)
8              (last; all functional phases must ship first)
```

Parallel opportunities summary:
- **2 ↔ 3** — different surfaces, no shared files. Two implementers can run simultaneously after 1B.
- **3A ↔ 3B** — within Phase 3, removals and side-rail additions are independent sub-tracks.
- **7 ↔ 2/3** — catalog reshape is self-contained after 1B.
- **5b ↔ 6/7** — once 5b lands, Phase 6 (completion gestures) and Phase 7 (catalog reshape, if not already shipped) can run in parallel with Phase 4 tutorial work.
- **5a ↔ 3** — different code regions; coordinate on `InspectorPanel` removal vs side-rail additions.

---

## D. Scope-changing facts surfaced during reconnaissance

These are facts that *changed* what the design docs assumed. The operator should be aware of each.

### D1. There is no `user_preferences` storage. (B4 — decided 2026-05-15)

The redesign docs assume `composer.default_mode` is "stored in the same table / model as other
user-level settings" (05-modes-and-opt-out.md §implementation-notes). **No such table exists.**

- `src/elspeth/web/auth/models.py` — `UserIdentity` / `UserProfile` are frozen dataclasses, not
  DB rows.
- `src/elspeth/web/sessions/models.py` — sessions store `id, user_id, auth_provider_type, title,
  created_at, updated_at, forked_from_*`. No mode field. The file does host the shared
  `MetaData()` and the existing tables (`sessions_table`, `chat_messages_table`, etc.).
- `src/elspeth/web/frontend/src/hooks/useTheme.ts` — theme preference uses **localStorage**. This
  is the only user-scoped preference precedent in the codebase.

**Operator decision 2026-05-15:** add a `user_preferences_table` to `sessions/models.py` alongside
the existing tables — "tear the bandaid off, it's going to happen eventually." Per
`project_db_migration_policy`, this is *not* a migration; the operator deletes the old sessions DB
when this lands. The Phase 1 plan implements this end-to-end.

### D2. Current "freeform default" is implicit. (Re A2)

Commit `82dd2e73b` ("default new sessions to freeform with switch-to-guided affordance") sets the
freeform default in the frontend by simply not calling guided-mode entry on session create. There
is no persisted setting today — the freeform default is a code-level decision in
`sessionStore.ts`'s `createSession`. The migration question A2 therefore reduces to: when Phase 1
ships and adds a real preference, what value does an existing user's first read see?

Recommended: read localStorage; if absent **and** the user has prior sessions → `freeform` (the
current implicit default); if absent **and** no prior sessions → `guided` (the new default for new
users); show the dismissible banner from 05-§discoverability when the heuristic surfaces freeform
for someone who hasn't explicitly set it.

### D3. Mode toggle infrastructure already exists in `ChatPanel.tsx`. (Re Phase 1 scope)

`ChatPanel.tsx` already imports `ExitToFreeformButton` and consumes guided session state. The
per-session toggle exists. Phase 1 does **not** rebuild it — Phase 1 only adds the
*account-default* layer above it.

### D5. B2 verdict (c) — Phase 5b adds a new audit table. (Resolved during Phase 5b planning)

`B2` is now resolved: **verdict (c) — new event class**, detailed in
[18-phase-5b-surface-llm-interpretation.md](18-phase-5b-surface-llm-interpretation.md). Phase 5b
adds an `interpretation_events_table` to the session audit DB. This is the third schema-extension
event under `project_db_migration_policy`: Phase 1A adds `user_preferences_table`; Phase 4 adds
`user_preferences.tutorial_completed_at`; Phase 5b adds `interpretation_events_table`; Phase 6
adds `composer_completion_events_table` (the fourth — adjudicated 2026-05-18 under Path A, see
[19a-phase-6a-backend.md](19a-phase-6a-backend.md) Task 1). The
cumulative DB-delete cost is acknowledged; the structural fix (migration runner) remains a follow-up
named in Phase 1A's review history.

### D6. Four schema additions ship without a migration runner (Phase 9 owns the fix)

Plans 12 (Phase 1A — user_preferences_table), 21a (Phase 4A — tutorial_completed_at
column), 18a (Phase 5b — interpretation_events_table + provenance enum extension),
and 19a (Phase 6 — composer_completion_events_table + append-only triggers, adjudicated
under Path A 2026-05-18) each add schema. Under project_db_migration_policy each
addition forces a DB delete, which wipes the *previous* phase's user state. Cumulative
effect: a user who completes the tutorial after Phase 4 ships gets the tutorial again
after Phase 5b ships, and again after Phase 6 ships.

The structural fix — a migration runner that preserves user-state tables across
schema changes — is OWNED BY A NEW PHASE 9 (post-launch). All four current schema
plans explicitly defer to Phase 9 with the caveat that pre-Phase-9 deploys to
production with real users are BLOCKED. Staging deploys are unblocked because
"WE HAVE NO USERS YET" per CLAUDE.md.

Phase 9 is added to the §B table below as IN PLANNING.

---

## E. What this roadmap deliberately does not do

- **Does not implement.** All plans now exist except Phase 4 (in progress). None are yet SHIPPED.
- **Does not adjudicate every question.** Section A is the operator's worksheet; resolved rows are
  marked VERDICT/RECOMMENDED. The remaining open question (B1) still requires fact-finding.
- **Does not estimate calendar time.** The CLAUDE.md memory `feedback_no_calendar_shipping_commitments`
  is explicit on this: ELSPETH ships work-until-done, not against dates.
- **Does not add scope.** The recommended calls hew to the design docs; the only *new* things since
  initial writing are the Phase 1 storage strategy (§D1) and the B2 event-class verdict (§D5).

---

## F. Memory references

All seven composer-design memory entries from the README plus, for this roadmap specifically:
- `project_db_migration_policy` — informs the (A) vs (B) storage call; three DB-delete events listed in §D5
- `feedback_no_calendar_shipping_commitments` — informs no-dates rule
- `project_composer_default_guided_with_opt_out` — supersedes the freeform-default commit, informs Phase 1 scope
- `project_composer_first_run_tutorial` — informs Phase 4 scope
- `project_composer_dynamic_source_from_chat` — informs Phase 5a scope

---

## G. Plan-quality review status

**REVIEWED + FIXED** (full review panel applied and changes incorporated):
- Phase 1A — [12-phase-1a-backend.md](12-phase-1a-backend.md)
- Phase 1B — [13-phase-1b-frontend.md](13-phase-1b-frontend.md)
- Phase 7 (all three parts) — [16a](16a-phase-7a-backend.md), [16b](16b-phase-7b-frontend.md), [16c](16c-phase-7c-frontend-integration.md)

**REVIEWED + FIXED** (added 2026-05-15, full review panel applied):
- Phase 2 — [14a](14a-phase-2a-backend.md), [14b](14b-phase-2b-frontend.md), [14c](14c-phase-2c-frontend-integration.md)
- Phase 3 — [15a1](15a1-phase-3a-removals-part-1.md), [15a2](15a2-phase-3a-removals-part-2.md), [15b1](15b1-phase-3b-side-rail-part-1.md), [15b2](15b2-phase-3b-side-rail-part-2.md)
- Phase 4 — [21a1](21a1-phase-4-backend-part-1.md), [21a2](21a2-phase-4-backend-part-2.md), [21b1](21b1-phase-4-frontend-part-1.md), [21b2](21b2-phase-4-frontend-part-2.md)
- Phase 5a — [17](17-phase-5a-dynamic-source-from-chat.md)
- Phase 5b — [18a](18a-phase-5b-backend.md), [18b](18b-phase-5b-frontend.md)
- Phase 6 — [19a](19a-phase-6a-backend.md), [19b](19b-phase-6b-frontend.md)
- Phase 8 — [20](20-phase-8-polish-and-telemetry.md)

**END-TO-END REVALIDATION 2026-05-16** (full panel re-run plus six cross-file
coherence checks per phase; 47 findings applied across 11 phases):
- All eleven phases CLEAR. Notable surfaced defects: per-instance-attribute
  AttributeError in Phase 7 source-quarantine inference (would have crashed
  every real source); 401-becomes-500 in Phase 1A anonymous-auth fixture;
  hallucinated import path in Phase 2 (`contracts.determinism` →
  `contracts.enums`); per-tab filter state regression in Phase 7c;
  `shareable_reviews_table` schema addition in Phase 6 (deleted under
  `project_db_migration_policy`); audit-write "telemetry-class" carve-out in
  Phase 6 (deleted under audit primacy).
- Cross-phase invariant verification:
  - §H1 cumulative DB-delete loop — Phase 9 stub drafted (see below); 1A/4A/5b
    each correctly defer to Phase 9 with production-deploy block.
  - §H2 AuditReadinessPanel mount handoff — verified end-to-end; 15b2 Task 9
    relocates the panel into SideRail.auditReadinessSlot (with App.tsx import
    + InspectorPanel.tsx import removal) BEFORE inspector deletion.
  - §H3 `_verify_session_ownership` extraction — verified; 14a extracts to
    `web/sessions/ownership.py`, `execution/routes.py` imports from there.
  - §H4 API client utility triplication — was a PHANTOM finding. Current
    `client.ts` has a single definition; no consolidation needed in Phase 8.
  - §H5 closed-enum ceremony — verified; Phase 5b correctly references
    `models.py:285-287`; Phase 6 dropped its `writer_principal` extension.
  - §H6 test pseudo-code placeholders — verified absent in 17, 18b, all
    Phase 4 plans.
  - §H7 module-level mutable state — verified; Phase 3A applies the closure
    pattern + `_resetForTests()` test hook.

**Phase 9 — PLAN READY (J1 APPROVED 2026-05-16):**
- Plan written 2026-05-16 at [22-phase-9-migration-runner.md](22-phase-9-migration-runner.md).
- J1 verdict: **(c) per-table preserve-on-recreate** — APPROVED by operator 2026-05-16.
  Rationale: best auditability (every action becomes a first-class Landscape
  event with table classification), matches the project's actual schema-
  change pattern (3-of-3 additive), uniform shadow-copy ceremony handles both
  add-column AND closed-enum-extension cases, lower ceremony than Alembic,
  refuses destructive changes loudly. Defensible cases for (a) SQL DDL diff
  and (b) Alembic-style preserved in the plan's trade-off tables at §2.
- Status: PLAN READY — full review panel applied and incorporated. Implementation
  gated on operator decisions #2 (post-ship rollback strategy) and #3
  (fixture-generation approach), plus Phase 1A and Phase 5b in git history.

Per operator instruction, all plans must clear review-and-fix before implementation. No plan has reached SHIPPED status yet. The existing-user migration is covered by the A2 row and §D2.

---

## H. Cross-phase findings — applied 2026-05-15

The full review panel (reality + architecture + quality + systems) on plans 2/3/4/5a/5b/6/8
surfaced cross-phase patterns now resolved or named:

1. **Cumulative DB-delete loop** — Phases 1A/4A/5b/6 each force a DB delete. Resolved by
   naming Phase 9 as the migration-runner owner (see §D6).
2. **AuditReadinessPanel mount-point ownership gap** — Phase 14c mounted in InspectorPanel;
   Phase 15b2 deletes the inspector. Resolved by assigning the relocation to 15b2 Task 9
   explicitly.
3. **`_verify_session_ownership` duplication** — between 14a's audit-readiness routes and
   execution/routes.py. Resolved by extracting to `web/sessions/ownership.py`.
4. **API client utility triplication** — `authHeaders`/`parseResponse` duplicated across
   preferences, auditReadiness, future API modules. Phase 8 consolidates.
5. **Closed-enum governance fragmentation** — multiple plans extend separate closed enums
   in parallel. Each plan now references the same `models.py:274-289` ceremony.
6. **Test pseudo-code bodies** — Plans 17 and 18b shipped with `/* ... */` placeholder
   assertions. Fixed in-place; check future plans for this pattern.
7. **Module-level mutable state without reset** — same shape as Phase 1A BLOCKER. Plan
   15a1's `lastValidatedVersion` fixed.
