# Phase 8 — Polish and Telemetry

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:subagent-driven-development` (recommended) or
> `superpowers:executing-plans` to implement this plan task-by-task. Steps
> use checkbox (`- [ ]`) syntax for tracking. This is the **final phase** of
> the UX redesign; it depends on every prior phase having shipped (or
> explicitly stubbed). Several tasks are **conditional** — they branch on
> whether the upstream phase landed the surface they polish. Do not skip
> the conditional probe step at the top of those tasks.

**Goal:** Close out the UX redesign. Wire deferred telemetry markers seeded
by earlier phases, emit mode-related aggregate metrics, replace the
generic template cards with audit-domain exemplars sourced from the
README, complete the session-sidebar → header-switcher migration, audit
and reorganise keyboard shortcuts, add a tutorial-replay affordance in
settings, run an accessibility audit across every new component, and
sweep the codebase for the dead code and CSS drift accumulated across
Phases 1-7. After this phase, the composer is at the full target state
described in [03-target-information-architecture.md](03-target-information-architecture.md);
telemetry tells the team whether the default-mode call (and the other
design bets) was right.

**Architecture:** Harvest of work seeded by every prior phase. No new
product capability beyond tutorial-replay. Conditional tasks probe
whether their upstream surface shipped; a missed probe is a documented
no-op (Phase 9 follow-up), never a silent degraded metric.

**Tech Stack:** Python/OTel (telemetry), React + Zustand + Vitest +
testing-library + axe-core (frontend), TypeScript.

**Sibling plans (must have shipped before this plan can be useful):**

- [12-phase-1a-backend.md](12-phase-1a-backend.md) — preferences endpoint; Task 1 wires its telemetry marker.
- [13-phase-1b-frontend.md](13-phase-1b-frontend.md) — `ComposerPreferencesPanel.tsx`; Task 6 mounts replay button here.
- [14a/b/c — Phase 2](14a-phase-2a-backend.md) — audit-readiness panel; Task 2 reads its aggregated state.
- [15a1/a2/b1/b2 — Phase 3](15a1-phase-3a-removals-part-1.md) — IA cleanup. Phase 3B introduces `HeaderSessionSwitcher`; Task 4 is conditional on it.
- **Phase 4** — Hello-world tutorial ([21-phase-4-hello-world-tutorial.md](21-phase-4-hello-world-tutorial.md)); Task 6 conditional on Phase 4's `tutorial_completed_at` column having shipped (see Phase 4 §"Cross-plan contract" for the PATCH-body shape Task 6 relies on).
- **Phase 5a** ([17-phase-5a-dynamic-source-from-chat.md](17-phase-5a-dynamic-source-from-chat.md)) / **Phase 5b** — Task 1 wires Phase 5a marker if present.
- **Phase 6** — Completion gestures (plan not yet written); Task 1 wires YAML-export marker if present.
- [16a/b/c — Phase 7](16a-phase-7a-backend.md) — catalog reshape; Task 3 follows its vocabulary and tone.

**Roadmap reference:** [00-implementation-roadmap.md](00-implementation-roadmap.md).
**Design reference:** [10-implementation-phasing.md](10-implementation-phasing.md) §"Phase 8".

---

## Pre-flight check (W1 — operator gate)

**Run BEFORE approving Phase 8 start.** Phase 8 is a harvest phase
whose surface is conditional on what earlier phases have shipped.
Approving Phase 8 without checking what's actually on disk risks
approving N tasks of work and delivering far less when probes miss.

Execute every Task 0 conditional probe at the operator-approval
moment and record outcomes. Probes to run (canonical set; Task 0
runs them again at execution time):

| Probe (run from `/home/john/elspeth`) | Hits if shipped | Gates |
|---|---|---|
| `grep -rn 'telemetry: deferred to Phase 8' src/elspeth/` | ≥1 | Task 1 Sub-tasks 7a (markers-harvest framing) |
| `grep -rn 'verify_token\|verify_share_token' src/elspeth/web/` | ≥1 | Task 1 Sub-task 7d (B3 cohort a) |
| `grep -rn 'auto_interpreted_opt_out' src/elspeth/web/` | ≥1 | Task 1 Sub-task 7e (B3 cohort b1) |
| `grep -rn 'audit_readiness\|composer/audit/readiness' src/elspeth/web/` | ≥1 | Task 1 Sub-task 7f (B3 cohort b2) |
| `ls src/elspeth/web/frontend/src/components/sessions/HeaderSessionSwitcher.tsx` | file exists | Task 4 (session-sidebar migration) |
| `grep -rn 'tutorial_completed_at' src/elspeth/web/` | ≥1 | Task 6 (tutorial-replay button) |

**Probe classification (S5 — load-bearing).** Pass-2 review caught
that the previous "≥75% / 50-74% / <50% auto-stop" rule treated all
six probes as equal weight, which they are not. The probes split into
two classes by what they gate:

- **Unconditional probes (would gate Phase 8 start if they existed):**
  **None.** No current probe gates an unconditional Phase 8 task.
  Tasks 0, 1 (markers harvest), 2 (mode metrics), 3 (templates), 5
  (shortcuts), 7 (a11y), and 8 (sweep) all run regardless of probe
  outcomes.
- **Conditional probes (miss = documented no-op for the gated
  sub-task, NOT a gate on Phase 8 start):** all six probes listed
  above. Every probe gates exactly one conditional sub-task; a miss
  routes that sub-task to its Case-B branch (documented no-op +
  Phase 9 follow-up), which is a fully-supported plan outcome rather
  than a stop condition.

**Operator decision rule (S5-reweighted):**

Because no probe gates an unconditional task, the old auto-stop rule
no longer applies. Instead the operator reviews probe outcomes and
decides scope for THIS Phase 8 wave:

- **Case A — scope IN:** for each missed probe, the operator decides
  whether to scope its conditional sub-task IN to the current wave
  anyway (deferring the actual emit to a follow-up commit once the
  upstream sibling ships) or OUT.
- **Case B — scope OUT for Phase 9:** the missed-probe sub-tasks
  are filed as Phase 9 follow-ups via the `21-phase-9-followups.md`
  cadence per §"Phase 9 follow-ups file — review cadence (S4 —
  load-bearing)".

The decision is a **scope conversation**, not an auto-stop. The
operator may also choose to split Phase 8 itself per §"Operator-gate
decisions (pass-2 outcomes)" Decision 1 (B3-r2 phase split) below
— that decision happens at the same gate and uses the same probe
outcomes as input.

**Edge case — all six probes miss.** Pre-S5 the rule would have
auto-stopped. Post-S5 the operator may still scope IN every
sub-task as a "wire the emit speculatively; document the no-op now;
the emit fires when the upstream sibling lands later" path, OR
scope every sub-task OUT to Phase 9. Both are valid; the
operator picks based on Phase 6/5b/2C delivery confidence.

**Record outcomes in `21-phase-9-followups.md`** at the gate so the
deferral decision is durable and discoverable, even when the
operator and Phase 8 implementer are different people.

This pre-flight is in addition to Task 0's runtime probes — the
operator gate happens once, before approval; Task 0's probes happen
again at execution, to catch any sibling-phase delivery between
approval and start.

---

## Operator-gate decisions (pass-2 outcomes)

Two decisions are explicitly deferred to the operator and MUST be
made before Phase 8 (or its first sub-phase) begins. Both arise from
the pass-2 architecture review; both have material consequences for
plan structure and scope; neither has a single defensible default
the plan author could pick on the operator's behalf.

### Decision 1 — B3-r2 phase split (architecture)

The plan's blast radius after pass-2 fixes is approximately **27
weighted units** (rough estimate: B1-r2 + B2-r2 + W8-r2 + Q-cluster
+ this pass's 12 fixes and 2 operator gates — each "fix" carries
~2-3 dependent edit sites under tracer-bullet discipline). That is
large for a single phase, particularly given that Phase 8 also
absorbs cross-phase emit ownership (B3 cohorts a/b1/b2) on behalf
of Phases 6 / 5b / 2C.

The architecture review recommends splitting Phase 8 into three
sub-phases with separate operator approvals:

- **Phase 8a — Preconditions (smallest, most-reversible-irreversibility):**
  - B1 audit-payload extension (adds `prior_trust_mode` key to the
    `trust_mode.changed` event payload; consumer-side updates in
    the audit panel and MCP tools per A2; co-land docstring on
    `proposal_events_table`).
  - B2 service-signature reshape on both
    `update_composer_preferences` service functions (atomic
    read-prior-record + write + return `(prior, current)`).
  - B2.b account-level scope narrowing baked into the helper-module
    shape (no `from_mode` on account-level helpers).
  - Phase 1A fixture co-land for both endpoints.
  - **No** telemetry emits, **no** new counters, **no** route-level
    changes.
  - Operator-gate the DB-delete on staging in isolation under this
    sub-phase — it's the only irreversible deploy step in 8a and
    it stands on its own merits.

- **Phase 8b — Telemetry harvest (compiles cleanly against 8a):**
  - Counter container extensions (Task 1 Step 5 in full).
  - `telemetry_phase8.py` helper module (Task 1 Step 4 in full).
  - W5 try/except wrapping on every new helper.
  - Route-level emits (Sub-tasks 7a, 7a', 7b, 7c, 7d, 7e, 7f).
  - The A9 tracer-bullet step at Task 1 Step 0 serves as 8b's
    natural validation gate — landing the tracer with 8a's
    preconditions already on disk is a much smaller integration
    risk than landing all three sub-phases at once.

- **Phase 8c — UX polish (independently revertable):**
  - Task 3 (templates → README audit-domain exemplars).
  - Task 4 (sidebar → header switcher).
  - Task 5 (keyboard shortcuts).
  - Task 6 (tutorial-replay button UI; the *emit half* of Task 6
    Step 7 actually belongs to 8b because it depends on the B2
    reshape — note this as a 8b/8c handoff in the operator
    decision rationale).
  - Task 7 (axe + jest-axe).
  - Task 8 (final sweep).
  - No audit/Tier-1 implications; reverts touch only UI surface.

**Resolution (2026-05-19, operator decision):** Option A — split
into 8a / 8b / 8c, operator approves three sub-phases separately.
The B1-r3 MeterProvider precondition (see §"MeterProvider
precondition (B1-r3 — load-bearing)" below) and the B4-r3
commit-msg hook installation (see §"Cohort attribution via commit
trailers (A4 — load-bearing)" below) both land inside 8a, which
formalises the split — there is now concrete infra content in 8a
that did not previously belong to the monolith. The DB-delete
remains isolated to 8a where it is the only irreversible operation.
The A9 tracer-bullet step at Task 1 Step 0 serves as 8b's natural
cross-sub-phase validation gate.

**Implementer guidance:** the per-task assignment above (8a vs 8b
vs 8c) is the load-bearing partition; treat the rest of this
document as the union of 8a + 8b + 8c. A successor refactor (Q13)
may emit three stripped sub-phase files plus an archived
`20-phase-8-review-history.md`, but this document remains the
canonical specification until that refactor lands.

### Decision 2 — S6 tutorial-replay boundary question

The `composer.tutorial.replayed_total` counter (Task 6 Step 7) is
currently treated as a non-decision-read superset exception per
§B2.b reasoning: no audit row, telemetry-only signal. But
tutorial-replay is a **user-write-intent** — the user clicked
"Replay hello-world tutorial" deliberately, the PATCH body says
`{"tutorial_completed_at": null}` (clearing Phase 4's
`tutorial_completed_at` column — see §"Cross-plan contract" in
[21a1-phase-4-backend-part-1.md](21a1-phase-4-backend-part-1.md)), and the
audit-relevant fact is the user's explicit intent to re-onboard.

CLAUDE.md's superset exception is named for non-decision **reads**
(read-path operational health: counter fires on a fetch failure
that has no decision content). The question pass-2 surfaces is
whether user-write-intent falls inside that exception, or whether
it merits a Landscape audit event under the primacy rule.

**Resolution (2026-05-19, operator decision):** Option C — defer
the boundary question to Phase 9. Phase 8 ships **without** the
`composer.tutorial.replayed_total` counter. Rationale: Option A
was found to violate CLAUDE.md audit primacy on pass-3 review —
the superset exception is named for non-decision **reads**
(read-path operational health, e.g. fetch failures with no
decision content), not for deliberate user-write-intent like the
replay click. Picking Option A would have required either
re-classifying the click as a read (semantically dishonest) or
amending CLAUDE.md to broaden the superset exception
project-wide (out of scope for Phase 8). Option B (audit-record
the replay) was correctly rejected by the original B2.b reasoning
and remains rejected. Option C is the least-commitment path that
keeps Phase 8 inside policy.

**Propagated strikes (Option C consequences):**

- Task 6 Step 7 (counter-emit half) is **scoped OUT of Phase 8**;
  the rest of Task 6 (UI, store, button, PATCH call, audit-side
  flag clear) **stays** in 8c.
- The §"Telemetry primacy explicit acknowledgment" bullet for
  `composer.tutorial.replayed_total` is rewritten as a Phase 9
  deferral pointer (not deleted — the name stays in the list so
  the deferral is discoverable from the canonical counter
  inventory).
- The counter container slot for `tutorial_replayed_total`
  (Task 1 Step 4 / Step 5) is removed.
- The OTel counter-list entry for `composer.tutorial.replayed_total`
  (A9 bootstrap step) is removed.
- Task 6 Q6 backend tests for the replay counter are removed;
  Task 6 retains only its UI / store / PATCH-call tests.
- `21-phase-9-followups.md` gains a new entry for the deferred
  counter; the boundary question is named there explicitly so
  Phase 9's design pass cannot drop it on the floor.

Sites that read "replay counter" or `tutorial_replayed_total`
in the body of this plan are now either Phase-9-followup pointers
or stale references that the propagated strikes have removed —
see commit history for the cohort.

---

## Scope boundaries

**In scope:**

- Wire every telemetry-deferral marker seeded by earlier phases (Task 1).
- Emit aggregate mode-related metrics: opt-out rate, completion rate,
  per-mode session-switch rate (Task 2). Aggregate-only, never
  user-attributable.
- Replace `TemplateCards.tsx` content with audit-domain exemplars
  sourced from the README "Example Use Cases" table (Task 3).
- Complete the session-sidebar → header-switcher migration; archive
  controls and a session filter on the new switcher (Task 4).
- Audit and reorganise keyboard shortcuts (App.tsx + ShortcutsHelp.tsx);
  group them by category (Task 5).
- Add a "Replay hello-world tutorial" button in the settings pane that
  nulls `user_preferences_table.tutorial_completed_at` via
  `PATCH /api/composer-preferences` with body
  `{"tutorial_completed_at": null}` (Task 6). **Conditional** on Phase 4
  having shipped the column. The PATCH contract is co-owned with Phase 4 —
  see §"Cross-plan contract — `tutorial_completed_at` PATCH semantics" in
  [21a1-phase-4-backend-part-1.md](21a1-phase-4-backend-part-1.md).
- Run axe-core against every new component from Phases 1-7; fix
  high-severity findings; file medium-severity findings as follow-ups
  (Task 7).
- Dead-code sweep, dead-comment sweep, lint clean-up, CSS variable
  consolidation across the new components from Phases 1-7. Final
  `npm run lint`, `mypy`, `pytest` run (Task 8).

**Out of scope (deferred to a successor cleanup phase — call it "Phase 9"
or surface to the operator as a follow-up):**

- Telemetry on **content** of user prompts (privacy boundary; only
  aggregate event counts cross the wire).
- Tutorial localisation. The hello-world tutorial copy is English-only.
- Performance work on any of the new components. The redesign assumes
  current frontend perf is acceptable; if axe-core reveals layout
  thrash, that's a Phase 9 issue, not a Phase 8 fix.
- Adding a "redo the tutorial" affordance to anywhere besides settings.
  Specifically: the empty-state chat does **not** get a "redo tutorial"
  link in Phase 8. If a user wants to redo the tutorial, they go to
  settings and click the button.
- Recipes / examples tab in the catalog (deferred per Phase 7 design).
- Catalog keyboard-shortcut reorganisation as a content change. Task 5
  only audits and regroups the existing shortcuts; it does not invent
  new ones for the catalog.
- The big-brother MCP tool (memory: `project_demo_big_brother_mcp_tool.md`)
  is **not** Phase 8 work; it sits behind §7.6 hardening.
- Frontend i18n for any of the new UI strings.
- **B3 cohort (c) — perf instrumentation in shipped phases.**
  `composer.audit.render_duration` (Phase 2C) and
  `composer.interpretation.resolve_duration` (Phase 5b) are pure
  perf signals; reopening shipped phases for perf instrumentation
  is out of proportion to the benefit and not security-relevant.
  Filed as Phase 9 follow-ups per §"Cross-phase telemetry — cohort
  split (B3 reshape)" → "Cohort (c) — Phase 9 follow-up filing".

**Explicit non-features:** Phase 8 does not add or remove any product
capability beyond the tutorial-replay button (everything else is
instrumentation, polish, accessibility, or content swap), and does not
retroactively change the wire contract for any earlier phase
(telemetry is emitted in-process; OTel exporter ships it).

---

## Caretaker logic policy

Phases 1B (banner), 4 (tutorial), and others add caretaker logic that
re-shows onboarding state when a DB delete wipes prior settings. Per
CLAUDE.md "WE HAVE NO USERS YET", that caretaker logic is currently
nullified by the delete-the-DB policy. Phase 8 does **not** retire
this logic — Phase 9 owns the migration-runner structural fix that
will give it permanent meaning. Phase 8 ensures caretaker logic is
wired correctly so when Phase 9 ships, the UX behaves as designed.

---

## Trust tier check

Per [CLAUDE.md](/home/john/elspeth/CLAUDE.md) the three-tier trust model
governs every boundary this phase touches. Each task below states which
tier its inputs and outputs live in. Summary up front:

**Tier 1 — Our data (audit / config / preferences DB):**

- `user_preferences_table.tutorial_completed_at` column (Task 6, read on
  bootstrap; cleared on retake): crash-on-anomaly per Phase 4's Tier-1
  read-side guard (`_row_to_prefs`). A non-NULL value that is not a
  `datetime` → `RuntimeError`. Defined and tested in
  [21a1-phase-4-backend-part-1.md](21a1-phase-4-backend-part-1.md) Task 3.

**Tier 3 — External data (source input):**

- **Tutorial-replay PATCH body**: Pydantic accepts `null` for the
  `tutorial_completed_at: datetime | None = None` field (Phase 4 contract);
  any non-`datetime`, non-`null` value is rejected at the boundary (422).
  `"yesterday"` (str), `false` (bool), `0` (int) all → 422. See Phase 4
  Task 4 for the route-level coverage.
- **README "Example Use Cases" content** (Task 3): build-time only —
  hand-curated into `TemplateCards.tsx`; no runtime Tier 3 boundary.

### Telemetry primacy explicit acknowledgment

Per [CLAUDE.md](/home/john/elspeth/CLAUDE.md) §"Telemetry and Logging"
the order is **audit first** (sync, crash-on-failure), **telemetry
next** (async, best-effort), **logging last** (only when audit and
telemetry are broken). Phase 8 adds telemetry **only** — no new audit
events, no new logging.

What this phase sends to the OTel meter:

- `composer.mode.opted_out_total` — counter, incremented inside the
  PATCH `/api/composer-preferences` route whenever the PATCH body sets
  `default_mode=freeform` (Task 2). Aggregate. No user ID, no session
  ID. **No attribute set** — post-state-only counter per
  §"Account-level scope narrowing (B2.b — load-bearing)". The
  `from_mode` attribute proposed in earlier drafts was dropped because
  the account-level preference write deliberately emits no audit event
  (see the "Operational signal only" module-level comment in
  `preferences/service.py`), so a transition-shaped counter
  would violate the superset rule.
- `composer.mode.opted_in_total` — counter, the symmetric post-state
  case, incremented whenever the PATCH body sets `default_mode=guided`.
  Also attribute-free per §B2.b.
- `composer.session.completed_total{mode, completion_verb}` — counter,
  fired by the completion-gestures emit-site in Phase 6 (Task 2 wires
  this if Phase 6 shipped). Attributes: `{mode: "guided" | "freeform",
  completion_verb: "save_for_review" | "run_pipeline" | "export_yaml"}`.
- `composer.session.switched_total{from_mode, to_mode}` — counter,
  fired whenever a user explicitly switches a session's mode mid-flow
  (Task 2). Attributes: `{from_mode, to_mode}` where both values are
  drawn from `src/elspeth/web/sessions/models.py:150`'s CHECK
  constraint on `trust_mode` (`'explicit_approve'` | `'auto_commit'`).
  **Not** `'guided'` | `'freeform'` — those are the account-level
  `default_composer_mode` vocabulary (see `models.py:1076`) used by
  the `opted_out` / `opted_in` counters above. The two vocabularies
  are intentionally distinct because the two DB columns measure
  different concepts (per-session trust-tier override vs. account-
  level default UX gesture); cross-vocabulary leakage is the B1-r2
  defect class — see §"Vocabulary discipline (B1-r2 — load-bearing)".
- `composer.tutorial.started_total` — counter slot declared in Phase 8
  (`sessions/telemetry.py`); emit site filled by Phase 4
  (`21a2-phase-4-backend-part-2.md` Task 8, when Phase 4 ships).
- `composer.tutorial.completed_total` — counter slot declared in Phase 8
  (`sessions/telemetry.py:317-323`); emit site filled by Phase 4
  (`21a2-phase-4-backend-part-2.md` Task 8, when Phase 4 ships) with a
  `completion_path` attribute taking values
  `first_time | skip | retake | repeat`. Phase 8's
  `record_tutorial_completed` helper (`telemetry_phase8.py:255-265`)
  is the attribute-free fallback; Phase 4 calls the counter directly
  with the `completion_path` attribute, so the helper is reserved for
  any Phase 8-internal use that does not want to set the attribute.
  (Phase 4 was originally specified with `complete_total` (no 'd');
  CR-1 — 2026-05-19 — realigned Phase 4 onto the already-shipped
  Phase 8 name to avoid a parallel namespace.)
- `composer.tutorial.replayed_total` — **DEFERRED to Phase 9** per
  the Decision 2 resolution above (Option C). The replay button
  ships in Task 6 8c; the counter does not. See
  `21-phase-9-followups.md` for the boundary question and the
  re-instrumentation work.
- `composer.phase_8.probe_failed_total{phase, probe}` — counter,
  fired by Task 0's probe-mechanism checks when an upstream phase
  probe returns "not found." Signals a conditional task that cannot
  run; ensures the absence is recorded rather than silently producing
  degraded metrics (see §Verification approach — Probe safety policy).
  Emitted via a module-local counter in `telemetry_phase8.py`
  (`_PHASE_8_PROBE_FAILED_COUNTER`), **not** via the
  `_SessionsTelemetry` container (per W8-r2 module-local counter;
  matches the existing `_PREFERENCES_PATCH_COUNTER` pattern that
  backs `composer.preferences.patch_total` in
  `preferences/service.py:85`).

What this phase **does not** send: prompt text, session IDs, user IDs,
audit-trail events, or any structlog/logger calls (zero new logging
statements; counter-emit failures surface via OTel exporter logs only).

#### How to read these counters (W4 — load-bearing)

Several counters above measure scope-distinct phenomena that look
similar at first glance. Downstream dashboards and product decisions
must respect the denominators:

**Semantic caveat (B3-r3 — load-bearing).** The `mode_changed=True`
attribute at `preferences/service.py:296` does **NOT** mean "the
value of `default_mode` differed from the prior state." It means
"the PATCH body included the `default_mode` field." A user who
PATCHes `default_mode=freeform` when the prior value was already
`freeform` fires `mode_changed=True` and increments both the
numerator (`opted_out_total`) and the denominator. The counter is
therefore a **set-rate**, not a **transition-rate**. The plan
deliberately did not change `mode_changed`'s semantics in Phase 8
(that's a Phase 9-or-later refactor) — the prose and dashboards
must instead match the actual semantic. Anyone labelling a chart
"opt-out rate (transitions)" using these counters is over-claiming.

| Counter | Denominator (correct ratio) | What it measures | What it does NOT measure |
|---|---|---|---|
| `composer.mode.opted_out_total` | `composer.preferences.patch_total{mode_changed=True}` (R6) — the **default-mode-field-present subset** of the existing account-level PATCH counter emitted from `update_composer_preferences` in `preferences/service.py`. The bare `composer.preferences.patch_total` is **NOT** the correct denominator: it increments on every PATCH including banner-dismissal-only PATCHes (per the existing counter site at `src/elspeth/web/preferences/service.py:86`), so a banner-heavy session would deflate the opt-out ratio. The existing emit attaches a `mode_changed` boolean attribute at `preferences/service.py:296` that is `True` whenever the PATCH body includes `default_mode` (regardless of whether the value differs from prior); filter on `mode_changed=True` to get the correct denominator. | **Set-rate**: fraction of account-level preference-PATCHes that include `default_mode` in the body and set it to `freeform`. Re-submitting the same `freeform` value increments both sides. | **Transition rate.** Does NOT measure the fraction of users who flipped *from* guided *to* freeform; a user re-PATCHing `freeform=freeform` inflates the metric without flipping. Also: not a per-user rate (a single user PATCHing repeatedly contributes multiple increments); not the fraction of all preference-PATCHes that opt out (that bare-`patch_total` ratio is depressed by banner-only PATCHes). |
| `composer.mode.opted_in_total` | `composer.preferences.patch_total{mode_changed=True}` (R6) — same denominator-correction reasoning as the opt-out counter above. | **Set-rate**: symmetric — fraction of account-level preference-PATCHes that include `default_mode` in the body and set it to `guided`. | Same caveats: not a transition-rate, not per-user. |
| `composer.session.switched_total{from_mode, to_mode}` | Total per-session preference-PATCHes (count session-level PATCHes against `/api/sessions/{id}/composer/preferences`) | Fraction of per-session preference-PATCHes that change `trust_mode` | Account-level default-mode changes (those fire `opted_out`/`opted_in` instead). |
| `composer.session.completed_total{mode, completion_verb}` | Total composer sessions reaching a terminal verb (Phase 6) | Distribution of completion verbs by mode | Aborts / non-completions (no terminal verb reached). |
| `composer.interpretation.opt_out_total` | Total interpretation-resolve attempts (Phase 5b) | Fraction of interpretation prompts the user opted out of | Per-user opt-out rate (same per-PATCH vs per-user caveat). |
| `composer.audit.fetch_failure_total` | `composer.audit.fetch_total` (if Phase 2C emits a request counter; otherwise an HTTP request count) | Fraction of audit-readiness fetches that fail | Whether the failure was user-visible (read-path failures may be silently retried). |

**No counter measures per-user-first-time-only events.** Achieving
per-user-first-time semantics requires a Tier-1 "has this user
opted out before" read against the preferences DB on every PATCH —
deliberately out-of-scope for Phase 8. If product needs that
metric, file as a Phase 9 follow-up; do not reinterpret
`opted_out_total` to mean it.

**Two account-level counters and one per-session counter are
deliberately routed through different endpoints** so that no single
PATCH increments multiple counters. Post-B2, the account-level
`mode_opted_out_total` / `mode_opted_in_total` fire only at
`PATCH /api/composer-preferences` (the account-level
`update_preferences` route handler in `preferences/routes.py`);
`session_switched_total` fires only at
`PATCH /api/sessions/{id}/composer/preferences` (the per-session
`update_composer_preferences` route handler in `sessions/routes.py`).
These are different HTTP
requests — a single user action increments at most one counter.

#### MeterProvider precondition (B1-r3 — load-bearing)

**Current state (verified 2026-05-19).** `src/elspeth/web/app.py:20`
imports `from opentelemetry import metrics` and several modules
(`composer/redaction_telemetry.py:12`, `composer/service.py:37`,
`composer/tools.py:30`, `blobs/service.py:14`, the existing
`preferences/service.py` counter site) call `metrics.get_meter()`,
but **no module in `src/elspeth/` ever calls
`metrics.set_meter_provider(...)`**. `opentelemetry-sdk` is in
`pyproject.toml` but never wired. With no provider set,
`get_meter()` returns OTel's default `NoOpMeter`, every
`counter.add()` call is a no-op, and every Phase 8 counter
(and every existing counter) emits to `/dev/null`. The W5
try/except wrap on Phase 8 helpers makes this **invisible by
design** — there is no exception to catch when the underlying
counter is a no-op.

**Why Phase 8 must fix it.** A9's Step 4 ("verify the counter
surfaces in the OTel exporter") cannot succeed against a NoOp
meter. The plan's tracer-bullet discipline collapses if the
end-to-end validation has no observable end. Shipping Phase 8 on
the current infra would land 12 new counters that look correct
in code review and pass every unit test (which assert against an
in-process `InMemoryMetricReader` fixture) but emit nothing in
production. That defeats the entire telemetry-harvest purpose
of Phase 8.

**Phase 8a MUST land before any 8b counter work** a minimal
in-process `MeterProvider` wired into the FastAPI app factory.
The smallest reasonable shape:

```python
# In src/elspeth/web/app.py, at module-import time (NOT inside
# create_app — the meter is process-global per OTel's design):
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import (
    OTLPMetricExporter,
)
from prometheus_client import make_asgi_app
from opentelemetry.exporter.prometheus import PrometheusMetricReader

_PROMETHEUS_READER = PrometheusMetricReader()
metrics.set_meter_provider(
    MeterProvider(metric_readers=[_PROMETHEUS_READER])
)
# Inside create_app(), mount the Prometheus scrape endpoint:
app.mount("/metrics", make_asgi_app())
```

**Discovery contract.** The endpoint `GET /metrics` on the running
web app must return Prometheus exposition format. The tracer
bullet's Step 4 (below) curls this endpoint and greps for the
expected metric name.

**Scope inside 8a.** This precondition lands as the **first**
8a task, before any of the B1 / B2 / B2.b preconditions begin.
Rationale: B1/B2/B2.b reshape `src/` for the *content* of the
counter; B1-r3 reshapes infrastructure so the counter is
*observable*. The B1-r3 fix is a one-file change (`app.py`),
adds two pyproject deps (`opentelemetry-exporter-prometheus`,
`prometheus_client`), and does not block on any of B1/B2/B2.b.

**Side effects (declare them now).** Wiring a real `MeterProvider`
turns on **every existing counter in the codebase**:
`composer.preferences.patch_total`,
`composer.redaction.*`, `composer.service.*`, `composer.tools.*`,
`blobs.*`. None of these have ever exported in production
because the provider was missing. The first deploy after 8a will
surface metrics that have been silently accumulating in the
binary's no-op meter — operators should be told to expect a
sudden "appearance" of counters they may have assumed were
already live. This is observability, not a regression; the
existing counter call sites are correct, they were just exporting
to `/dev/null`. Document this in the 8a deploy notes.

**Out of scope for 8a (filed as Phase 9 followups during 8a
landing):** OTLP exporter to a remote collector, AzureMonitor
integration, sampling/aggregation policy beyond OTel defaults,
metric-naming audit across all existing call sites, dashboards.
Phase 8a ships an exposition endpoint; Phase 9 ships the rest of
the pipeline.

#### Audit-payload precondition (B1 — load-bearing)

The `from_mode` / `to_mode` attributes on the counters above describe a
**transition**. For telemetry to be a strict subset of audit, the
corresponding audit row must record **both** the prior state and the
new state. Recording only the post-state in audit while telemetry
emits the transition makes telemetry a superset of audit-recorded
reality, which violates the primacy order.

Current gap: `service.update_composer_preferences` (in
`/home/john/elspeth/src/elspeth/web/sessions/service.py`) writes an
`event_type="trust_mode.changed"` row into `proposal_events_table`
whose JSON `payload` records only the new `trust_mode` (and
`density_default`). The prior `trust_mode` is not captured. The audit
row must be extended to record `prior_trust_mode` alongside
`trust_mode` in the same payload (and equivalently for any other
telemetry attribute Phase 8 adds that captures a transition rather
than a post-state).

Ordering: this audit-payload extension is a **precondition** for
Task 1 Sub-task 7a. Phase 8 must co-land the audit-payload extension
and the telemetry emit in the same commit, **or** land the audit
extension first as a standalone commit that precedes the emit wiring.
The emit must never land before the audit extension; the audit-primacy
ordering rule (audit fires first, telemetry next) is non-negotiable.

Schema-cohort note: the audit-payload extension is itself a Tier-1
schema concern per [CLAUDE.md](/home/john/elspeth/CLAUDE.md) §"Three-
Tier Trust Model" and the project's "DB migration = delete the old
DB" policy. Although `proposal_events_table.payload` is a JSON column
(so no DDL column-add is required for the field itself), the *shape*
of the JSON payload is a contract that consumers (audit panel, MCP
analysis tools, replay tools) read by key. Changing the contract is a
schema-change cohort: the operator must delete the old sessions DB on
next deploy so prior rows do not appear with the new schema's keys
absent. This precondition is cross-referenced from Task 1 Sub-task 7a
and from the §Risks table.

**Downstream-reader audit (A2 — load-bearing).** Before B1 lands,
the implementer MUST grep for every downstream reader of the
`trust_mode.changed` payload shape and confirm each handles the
post-extension payload correctly (extra `prior_trust_mode` key
present alongside the existing `trust_mode` and `density_default`
keys). Known consumers per the plan archive:

- The audit panel (the Phase 2C `AuditReadinessPanel` family —
  search `src/elspeth/web/frontend/src/components/audit/` for
  reads of `payload.trust_mode`).
- The Landscape MCP analysis tools — specifically
  `elspeth-mcp explain_token` (re-reads payload by key in the
  lineage trace; verify against `src/elspeth/mcp/` or the
  equivalent server-tool implementation).
- Any replay tools that walk `proposal_events_table` rows
  (`grep -rn "proposal_events_table\|trust_mode.changed" src/elspeth/`).

For each consumer found, verify: (a) it does not assert the payload
has *exactly* the legacy key set (e.g., a `len(payload) == 2` check
would silently flip from passing to failing once `prior_trust_mode`
appears); (b) it does not crash on the new key under
crash-on-anomaly Tier-1 reads (CLAUDE.md §"Tier 1: Our Data" — any
unexpected key in our own data crashes immediately, so a Tier-1
read that enumerates expected keys must include `prior_trust_mode`
in the expected set after B1 lands).

Co-land the consumer-side updates in the same commit as the B1
audit-payload extension, OR confirm at grep time that no
consumer enumerates the payload keys defensively (the bare-key-
access pattern `payload["trust_mode"]` survives extension; the
key-set-enumeration pattern does not).

Additionally, co-land a docstring at the `proposal_events_table`
definition in `src/elspeth/web/sessions/models.py` that documents
the payload contract explicitly. The docstring MUST enumerate every
key currently in scope, when it was added, and which phase owns
each:

```python
# Payload contract for event_type="trust_mode.changed":
#   trust_mode: str — the new value the PATCH set
#       (vocabulary: 'explicit_approve' | 'auto_commit' per the
#       CHECK constraint on sessions_table.trust_mode; see
#       models.py:150).
#   prior_trust_mode: str — the value the PATCH overwrote
#       (added Phase 8 B1; same vocabulary as trust_mode above;
#       required for the per-session session-switched telemetry
#       counter to satisfy the audit-primacy superset rule).
#   density_default: str — the new density_default value
#       (vocabulary per the column's CHECK constraint).
# Adding a new key here is a Tier-1 schema-cohort change (per
# CLAUDE.md "DB migration = delete the old DB"); document the
# key, its vocabulary, and the owning phase at the same time.
```

Without the docstring, the JSON-payload key contract is implicit
and the next contributor extending it has no way to discover the
contract without grepping for every reader site individually.

The **superset rule** holds: every operational signal Phase 8 emits is
already represented in the audit trail (session events, preference
PATCH events). Telemetry is a strict subset of audit-recorded reality.

**Channel decision — no frontend telemetry primitive in Phase 8.** Every
counter listed above emits from a backend handler (the `composer-preferences`
PATCH route, the session-mutation routes, the completion-gesture routes
that Phase 6 already audits). Phase 3A's and Phase 1B's deferred
"frontend telemetry breadcrumb" notes (R4: locate by keyword search
`grep -n 'frontend telemetry\|telemetry breadcrumb' docs/composer/ux-redesign-2026-05/15a1-phase-3a-removals-part-1.md docs/composer/ux-redesign-2026-05/15a2-phase-3a-removals-part-2.md` —
line numbers are intentionally not cited because earlier inline citations
went stale after pass-1 reshapes; the keyword search is the durable
locator) are resolved **by emitting from the backend route the
frontend action invokes**, not by adding a frontend telemetry module.
Rationale: the backend already audits the action; emitting an OTel
counter at the same site preserves audit primacy without introducing a
second emission channel that could drift from the audit record (the
superset rule above). If a future operational signal genuinely requires
frontend-only emission (e.g., a UI interaction the backend never sees),
that decision belongs to a successor phase with its own primacy
analysis; Phase 8 does not establish a frontend telemetry primitive.

#### Service signature precondition (B2 — load-bearing)

The B1 precondition makes the audit payload record both prior and
new state. For the telemetry emit to read those values, the route
handler must have the prior state **in scope**. It does not, today.

Current gap: both `update_composer_preferences` service functions
have the same defect.

- Account-level:
  `preferences_service.update_composer_preferences` in
  `/home/john/elspeth/src/elspeth/web/preferences/service.py`
  (called from the `update_preferences` route handler in
  `web/preferences/routes.py`) writes the new
  default-mode row but does not read the prior row and does not
  return the prior value. The account-level route therefore has no
  `prior.default_mode` in scope.
- Per-session:
  `sessions_service.update_composer_preferences` in
  `/home/john/elspeth/src/elspeth/web/sessions/service.py`
  (called from the per-session `update_composer_preferences` route
  handler in `web/sessions/routes.py`) has the same shape: it
  writes the new `trust_mode` row but does not load the prior record
  and does not return both old and new. The per-session route
  therefore has no `prior.trust_mode` in scope.

Without the prior value in scope, the route handlers cannot compute
the transition predicates that Task 2 Step 3
(`if prior.trust_mode != current.trust_mode`) and Task 6 Step 7
(`if prior.tutorial_completed_at is not None and current.tutorial_completed_at is None`)
depend on, and cannot compile the `from_mode=prior.trust_mode`
argument that Task 2 Step 3 passes to `record_session_switched`.
Sub-task 7a does not consume `prior.*` after B2.b (see
§"Account-level scope narrowing (B2.b — load-bearing)" below) but
lands the symmetric reshape for code-shape consistency with the
per-session function and to leave the seam open for the
future-promotion path described in §B2.b.

Fix: reshape **both** service functions to load the prior record
**inside the same transaction** as the write (no TOCTOU window
between read and write) and return a typed result that exposes both
`prior` and `current` (e.g., a small frozen dataclass with two
attributes, or a `(prior, current)` tuple). The route handlers then
unpack the result and have both values in scope at the emit site.

Ordering: this service-signature reshape is a **precondition** for
**two** emit sites: Task 1 Sub-task 7a (account-level opt-out /
opt-in emit) and Task 2 Step 3 (per-session session-switched emit).
Task 6 Step 7 (the tutorial-replay counter emit, which would
formerly have been the third dependent) was deferred to Phase 9
by the Decision 2 resolution (Option C — see §"Decision 2"
above), so the B2 reshape no longer carries it as a dependent
inside Phase 8. The reshape is still required for Task 6's UI
(`bootstrap()` reads `tutorial_completed_at` and the button PATCHes
`{"tutorial_completed_at": null}` via the existing endpoint), but
the *route-handler* consumer
of `prior` is reduced to the two cases above. Either both emits
land after the reshape, or each emit co-lands with the reshape of
its own service function in the same commit. The reshape must
never lag the emit; landing the emit first would not compile.

Test-fixture ripple: Phase 1A established route- and service-level
test fixtures for both endpoints. Changing the service-function
return shape changes what those fixtures consume. The fixture
updates **must co-land in the same PR** as the service-signature
change — otherwise unrelated tests will break in CI and the failure
will be misattributed to whichever commit happened to expose it.
Co-land or block: do not split the service reshape and the fixture
update across PRs.

**Option not taken — read-before-write from the route handler.** A
naive alternative is to have each route handler call
`get_*_preferences` immediately before the PATCH to surface the
prior state, leaving the service signatures unchanged. This is
rejected:

- It opens a TOCTOU race between the read and the write — a
  concurrent PATCH could land between the two calls and the
  recorded `prior` would not match what the write actually
  overwrote. The audit trail would then contain a "transition"
  that never happened.
- It adds a second DB round-trip per PATCH for a value the write
  transaction already has at hand inside its own session.
- It pushes Tier-1 atomicity reasoning into the route layer, where
  it does not belong — the service is the right layer to keep
  read-and-write atomic.

Future implementers must not pick this path. The atomic
read-write-return reshape is the only acceptable form.

**Option not taken — additive seam (new sibling method).** A second
naive alternative is to introduce a new sibling method on each
service (e.g.,
`update_composer_preferences_with_prior(...) -> (prior, current)`)
and leave the existing `update_composer_preferences` untouched, so
existing call sites keep their current return shape and only the new
Phase 8 emit sites consume the new method. This is also rejected
(A3):

- It doubles the service surface area for a single behavioural
  outcome — two methods that do almost the same thing diverge over
  time and create call-site selection ambiguity ("which one should
  I call?").
- It leaves a deprecated `update_composer_preferences` requiring
  eventual removal. Per [CLAUDE.md](/home/john/elspeth/CLAUDE.md)
  §"No Legacy Code Policy", deprecation shims and "both old and
  new" branches are explicitly forbidden. The project's discipline
  is to change all call sites in the same commit, not to add
  parallel paths.
- The Phase 1A test-fixture ripple still happens eventually — the
  additive seam just defers the cost rather than avoiding it,
  while the project incurs the divergence risk in the interim.

The chosen replacement approach (atomic reshape of both service
functions + Phase 1A fixture co-land in the same PR) is higher
up-front cost but lower total cost across the project's "no legacy
code" lifecycle. Future implementers must not pick the additive
seam.

**Account-level scope narrowing (B2.b — load-bearing).** The B2
reshape applies to **both** service functions for symmetry, but
the symmetric telemetry payload does **not** apply to both. The
account-level `preferences_service.update_composer_preferences`
emits a deliberate architectural decision documented in the
"Operational signal only" module-level comment in
`/home/john/elspeth/src/elspeth/web/preferences/service.py`:
"Operational signal only — preferences are user state, not a
pipeline decision boundary, so NO Landscape emit… If a future
phase wires a preference into an execution boundary (e.g.
trust_mode gating auto-commit), promote this to a Landscape emit
at that moment — the counter is the seam." Today the function
writes an OTel counter (`composer.preferences.patch_total`) and
no audit event. Phase 8 respects this prior architectural choice.

Resulting telemetry-shape adaptation:

- **Per-session signal** (`composer.session.switched_total`, fired
  from `sessions/routes.py`) keeps its `{from_mode, to_mode}`
  attributes. The per-session route already writes a
  `trust_mode.changed` audit event into `proposal_events_table`;
  the B1 audit-payload extension on **that** event satisfies the
  superset rule for this counter.
- **Account-level signals** (`composer.mode.opted_out_total` and
  `composer.mode.opted_in_total`, fired from
  `preferences/routes.py`) DROP the `from_mode` attribute.
  They become **post-state-only** counters: increment
  `composer.mode.opted_out_total` unconditionally whenever the
  PATCH body sets `default_mode=freeform`; increment
  `composer.mode.opted_in_total` whenever the PATCH body sets
  `default_mode=guided`. No transition information is encoded
  in the counter, so no transition information needs to be in
  an audit row, so the superset rule holds vacuously.

What changes vs design doc 10's "Phase 8" intent: design doc 10
asked for an "opt-out rate". That rate is still measurable as
`composer.mode.opted_out_total / composer.preferences.patch_total{mode_changed=True}`
(a post-state ratio, with the `mode_changed=True` denominator
correction per §"How to read these counters (W4 — load-bearing)"
and its B3-r3 semantic caveat — bare `patch_total` is **NOT** the
correct denominator because banner-dismissal PATCHes deflate it,
and even the `{mode_changed=True}` filter measures a *set-rate*
on the `default_mode` field rather than a transition-rate). What
is lost is the **per-from-state breakdown** — Phase 8 cannot
answer "of users who opted out, how many were on `guided` vs
`unknown`?" without an audit row recording the prior state. The
product team accepted this loss of fidelity as the cost of
preserving the prior architectural decision; the alternative is
described below.

**Option not taken — promote account-level preferences to audit.**
The naive alternative is to wire `update_composer_preferences`
(account-level) to write a new audit event into a new
preferences-audit table (or extend an existing table) so
`prior_default_mode` is recorded and the counter can keep its
`from_mode` attribute. This is explicitly rejected for Phase 8:

- It contradicts the existing "Operational signal only"
  module-level comment in `preferences/service.py` that was made
  deliberately by a prior architect. Overriding such a decision
  is a senior-design concern, not a polish-phase task.
- The audit-schema choice is itself a Tier-1 decision per
  [CLAUDE.md](/home/john/elspeth/CLAUDE.md) §"Three-Tier Trust
  Model" and the project's "DB migration = delete the old DB"
  policy. Introducing a new audit table (or extending the
  preferences row shape into a new audit row) requires operator
  approval of a DB-delete on next deploy. Phase 8 is polish, not
  schema change.
- Phase 8's design doc never asked for account-level audit. The
  design-doc-10 intent is satisfied by the post-state ratio; the
  per-from-state breakdown is not a stated requirement.

Future implementers may revisit this if account-level preferences
are wired into an execution boundary (the seam the source-code
comment identifies). At that moment, the architectural decision
flips and the counter regains its `from_mode` attribute as part
of the same change.

Cohort cross-reference: this scope narrowing affects Sub-task 7a
(account-level emit; reshapes from transition pseudocode to
post-state pseudocode) and the helper signatures in
§"Telemetry primacy explicit acknowledgment" and Task 1 Step 4
(account-level helpers drop `from_mode`; `record_session_switched`
retains it). Task 6 Step 7 (the tutorial-replay counter emit
that would formerly have been a third reference here) was deferred
to Phase 9 by the Decision 2 resolution above; the B2.b
telemetry-shape question for that counter is recorded in
`21-phase-9-followups.md` for Phase 9's design pass.

**Vocabulary discipline (B1-r2 — load-bearing).** The account-level
and per-session preference columns use **distinct value
vocabularies**:

- Account-level `default_composer_mode IN ('guided', 'freeform')`
  (`src/elspeth/web/sessions/models.py:1076`).
- Per-session `trust_mode IN ('explicit_approve', 'auto_commit')`
  (`src/elspeth/web/sessions/models.py:150`).

Neither column allows NULL or `'unknown'`. This is a long-standing
schema decision — the two columns measure different concepts
(default UX gesture vs. per-session trust-tier override) and have
always been linguistically and semantically separate.

The pass-2 plan review caught a pass-1 draft that defined a single
`_ModeName = Literal["guided", "freeform", "unknown"]` shared across
the account-level helpers (`record_mode_opted_out`,
`record_mode_opted_in`) **and** the per-session helper
(`record_session_switched`). Because Task 2 Step 3 fires
`record_session_switched(from_mode=prior.trust_mode,
to_mode=current.trust_mode)`, the runtime values would be
`'explicit_approve'` / `'auto_commit'` — outside the shared
Literal's set. `_assert_mode` would raise `ValueError`; the W5
swallow wraps deliberately **after** the assert (so input-validation
defects still crash loudly, per offensive programming); the
exception would escape the helper, the audit row would already have
committed per B1 ordering, and the user PATCH would 500 **after**
the audit row stood. That inverts audit primacy at exactly the
surface Task 2 instrumented. The fabricated `'unknown'` sentinel
was a parallel defect — it appeared in the test parametrise list
without anyone checking that the column CHECK constraint admits it.

Resolution: the helper module now distinguishes the two vocabularies
with separate types. The per-session helper uses
`_SessionTrustMode = Literal["explicit_approve", "auto_commit"]`
and `_assert_session_trust_mode`. The account-level helpers are
post-state-only per §B2.b above and take no mode kwarg — they
need no Literal at all, so the shared `_ModeName` / `_KNOWN_MODES`
identifiers are deleted from the helper module entirely. See Task
1 Step 4 for the canonical helper-module shape.

Future telemetry additions touching either column MUST validate
against the correct CHECK constraint. Cross-vocabulary leakage
produces silent ValueError-then-500 cascades because the W5
swallow deliberately wraps after `_assert_*`. The Task 1 Step 3
regression test for `record_session_switched` includes
`from_mode="guided"` (a value that is *valid* for the account-level
vocabulary and *invalid* for the per-session vocabulary) as the
canonical cross-vocabulary-leak assertion.

---

## Cross-phase telemetry — cohort split (B3 reshape)

**Honest framing (B3 — load-bearing).** Earlier drafts of this
section listed six metrics under a "minimum instrumentation bar" and
asserted that prior phases "should have emitted" them. Direct
inspection of Phases 2C, 5b, and 6 plans shows that **none** of the
six is actually named in any sibling phase. Phase 8 cannot "verify
their presence" — it would be creating a requirement against sibling
phases that have already shipped (2C, 5b) or are mid-delivery (6),
without those siblings being told. That framing is dropped.

After review, the six metrics split into three cohorts with three
different appropriate resolutions. Phase 8 owns cohorts (a), (b1),
and (b2). Cohort (c) defers to Phase 9 follow-ups.

**Cohort attribution via commit trailers (A4 — load-bearing).** Phase
8 absorbs ownership of emits whose natural home is Phases 6, 5b, and
2C (cohorts (a)/(b1)/(b2) respectively). When a future contributor
runs `git blame` on `src/elspeth/web/shareable_reviews/`,
`src/elspeth/web/audit_readiness/`, or the Phase 5b opt-out route,
they will see a Phase 8 commit editing code that "belongs" to a
different phase, with no inline explanation of why. To make the
absorption discoverable:

- Every commit that lands a cohort-(a) emit (Sub-task 7d) MUST
  include the trailer `telemetry-backfill: shareable-reviews`.
- Every commit that lands the cohort-(b1) emit (Sub-task 7e) MUST
  include the trailer `telemetry-backfill: interpretation-opt-out`.
- Every commit that lands the cohort-(b2) emit (Sub-task 7f) MUST
  include the trailer `telemetry-backfill: audit-readiness`.

The trailer is the read-time seam: a future maintainer doing `git
blame` and then `git log -1` on the offending commit immediately
sees the cross-phase attribution and can navigate to this section
for the full rationale. Without the trailer, the absorption is
mute and the cross-phase coupling re-becomes invisible debt.

**B4-r3 enforcement (load-bearing).** The pass-3 review caught
that the "MUST" above was unenforced — no commit-msg hook, no CI
grep — and the project's `cicd-allowlist-audit` memory documents
what happens to unenforced conventions (51% growth in 32 days on
the L3 allowlist). Phase 8a therefore lands a **commit-msg hook**
that mechanically rejects commits violating the rule:

- **Hook file:** `.githooks/commit-msg-telemetry-backfill` (new
  file in 8a) — runs against `$1` (the commit message path that
  git passes to commit-msg hooks).
- **Trigger condition:** the commit touches at least one path
  under any of the cohort directories:
  - cohort (a): `src/elspeth/web/shareable_reviews/`
  - cohort (b1): the Phase 5b interpretation-opt-out route
    (locate at 8a authoring time — likely under
    `src/elspeth/web/composer/` or `src/elspeth/web/sessions/`;
    cite the actual file in the hook source)
  - cohort (b2): `src/elspeth/web/audit_readiness/`
- **Rule:** if the commit message body does **not** contain the
  stable trailer token required by the touched cohort, exit 1 with a
  message naming the cohort directory that
  triggered the check, the missing trailer, and the
  §"Cohort attribution via commit trailers" section to read.
- **Installation:** 8a's first task (alongside MeterProvider
  wiring) is to add `core.hooksPath = .githooks` to the repo's
  `.git/config` via `git config --local core.hooksPath .githooks`,
  mark the hook executable (`chmod +x`), and document in the 8a
  README that fresh clones must run the same `git config` line
  (the project does not use Husky or similar auto-installers).
- **CI backstop:** 8a also lands a CI step that runs the same
  predicate against every commit in the PR diff (`git log
  --format='%B' main..HEAD` piped through the same regex check
  on commits whose `git show --name-only` includes a cohort
  path). The CI backstop catches the case where a contributor
  has not installed the local hooks. This is the same belt-and-
  braces pattern the project uses for `enforce_tier_model.py`
  (local pre-commit + CI gate).
- **Allowlist file (`config/cicd/enforce_telemetry_backfill_trailer/`):**
  follow the existing `enforce_tier_model` shape — one entry per
  file path where a cohort-touching commit was made *before* the
  hook landed (there should be zero such entries at 8a landing;
  the allowlist exists so future legitimate exceptions have a
  controlled escape valve and so the convention can be ratcheted
  rather than swing-broken when a refactor moves a cohort
  directory).

The hook is the mechanical enforcement; the prose above remains
the *why*. Removing the hook is a Phase 9-or-later decision that
must explicitly cite this section.

**Path not taken — push the cohort emits back as subtickets in
Phases 6 / 5b / 2C.** A reviewer might reasonably ask why Phase 8
absorbs the emits instead of filing subtickets against the owning
phases. The decision is an explicit operator gate — see §"Operator-
gate decisions (pass-2 outcomes)" Decision 1 (B3-r2 phase split)
below. The current §"Cross-phase telemetry — cohort split (B3
reshape)" text presumes Phase 8 ownership; if the operator chooses
the alternative split, this section is reshaped under the
operator's direction at that gate.

| Cohort | Metric | Owner | Phase 8 task | Notes |
|---|---|---|---|---|
| (a) | `composer.share.token_verify_failure_total` | Phase 8 (was: Phase 6) | NEW Task 1 Sub-task 7d | Wires at Phase 6's verify-failure site. Telemetry-only signal: verify-failure is a non-decision read-path failure (no audit row at the verify site per 19a §"Audit-event recording" — the verify-failure branch is not in the enumerated audit-events list; 19a:117 "No DB round-trip on verify" supports the absence indirectly because no DB session implies no audit row could have been written, R5); CLAUDE.md superset exception applies. No new audit event. Conditional on Phase 6's token-verify path having shipped (probe in Task 0). |
| (a) | `composer.share.link_expiry_hit_total` | Phase 8 (was: Phase 6) | NEW Task 1 Sub-task 7d | Wires at Phase 6's expiry-hit site. Telemetry-only signal: expiry-hit is a non-decision read-path failure; superset exception applies. No new audit event. Same probe; same conditional. |
| (b1) | `composer.interpretation.opt_out_total` | Phase 8 (was: Phase 5b) | NEW Task 1 Sub-task 7e | Wires at Phase 5b's opt-out route. Audit-derivable from `interpretation_source='auto_interpreted_opt_out'` rows in the Landscape; superset rule satisfied. |
| (b2) | `composer.audit.fetch_failure_total` | Phase 8 (was: Phase 2C) | NEW Task 1 Sub-task 7f | Wires at Phase 2C's audit-readiness fetch site. Telemetry-only signal: the underlying event is a non-decision read failure on the audit-readiness endpoint, so the CLAUDE.md superset exception for non-decisions applies. No new audit event. |
| (c) | `composer.audit.render_duration` | **Phase 9 follow-up** | (none — filed in `21-phase-9-followups.md`) | Pure perf signal. Reopening shipped Phase 2C for perf instrumentation is expensive and not security-relevant. |
| (c) | `composer.interpretation.resolve_duration` | **Phase 9 follow-up** | (none — filed in `21-phase-9-followups.md`) | Same — perf-only, no security impact, defer. |

### Probe handling for cohort (a)

Phase 6 is mid-delivery as of Phase 8's authoring. When Phase 8
fires, Task 0 must probe whether Phase 6's `ShareableReviewService`
token-verify path has shipped (exact symbol name — `verify_token`,
`verify_share_token`, or equivalent — locate at execution time):

- **Case A — probe hits:** Phase 6 has shipped; wire the two
  cohort-(a) counter emits at the verify-failure and expiry-hit
  branches per Sub-task 7d.
- **Case B — probe misses:** Phase 6 lands during or after Phase 8.
  Sub-task 7d is a **documented no-op**, the two counters remain in
  the container (added in Step 5 below; unused counters cost
  effectively nothing), and the wiring is filed as a Phase 9
  follow-up. Operator surface required per §"Probe safety policy"
  so the Phase 6/8 timing collision is visible at close.

### Counter container additions (B3)

Cohorts (a), (b1), and (b2) introduce four new Phase-8-owned counters.
They need slots in `_SessionsTelemetry` (the existing counter
container in `src/elspeth/web/sessions/telemetry.py`):

- `share_token_verify_failure_total` (cohort a)
- `share_link_expiry_hit_total` (cohort a)
- `interpretation_opt_out_total` (cohort b1)
- `audit_fetch_failure_total` (cohort b2)

These extensions are listed under Task 1 Step 5 alongside the
existing B1+B2 additions; cross-reference there for the canonical
counter-name strings (the `composer.share.*`, `composer.interpretation.*`,
`composer.audit.*` form follows the existing convention from the
container).

### Cohort (c) — Phase 9 follow-up filing

`composer.audit.render_duration` (would have been Phase 2C) and
`composer.interpretation.resolve_duration` (would have been Phase 5b)
are pure perf signals. They are valuable but reopening shipped phases
for perf instrumentation is out of proportion to the benefit. Per
CLAUDE.md "WE HAVE NO USERS YET" the cost of deferring perf
observation is low; per project convention, perf phases are tracked
separately from product-correctness phases.

Phase 8 will file both metrics as entries in
`docs/composer/ux-redesign-2026-05/21-phase-9-followups.md` (created
lazily by Task 0; this is the first known content for that file).
Each entry names the owning phase (2C / 5b), the metric name, the
emit site to consider, and the explicit rationale that Phase 8
declined the scope.

---

## File structure

**Files this plan creates:**

```text
src/elspeth/web/composer/telemetry_phase8.py
src/elspeth/web/composer/telemetry_phase8_test.py
src/elspeth/web/frontend/src/components/chat/templates_data.ts
src/elspeth/web/frontend/src/components/chat/templates_data.test.ts
src/elspeth/web/frontend/src/components/settings/TutorialReplayButton.tsx
src/elspeth/web/frontend/src/components/settings/TutorialReplayButton.test.tsx
src/elspeth/web/frontend/src/test/a11y/axe-config.ts
src/elspeth/web/frontend/src/test/a11y/setup.ts
src/elspeth/web/frontend/src/test/a11y/components.a11y.test.tsx
docs/composer/ux-redesign-2026-05/21-phase-9-followups.md     (only if any follow-ups accumulate)
```

**Files this plan modifies:**

- `src/elspeth/web/sessions/routes.py` (Task 2 Step 3: per-session
  session-switched telemetry emit on
  `PATCH /api/sessions/{session_id}/composer/preferences`; depends on
  the B2 service reshape below)
- `src/elspeth/web/sessions/service.py` (Task 1, 2: counter helpers;
  and B2: reshape `update_composer_preferences` to load the prior
  record atomically and return both prior and current — precondition
  for the per-session session-switched telemetry emit. See
  §"Service signature precondition (B2 — load-bearing)".)
- `src/elspeth/web/sessions/service.py` (B1 audit-payload extension —
  record `prior_trust_mode` alongside `trust_mode` in the
  `trust_mode.changed` audit event before telemetry wires up; see
  §"Audit-payload precondition (B1 — load-bearing)" and Task 1
  Sub-task 7a precondition)
- `src/elspeth/web/sessions/models.py` (B1 — the audit-event payload
  is a JSON column on `proposal_events_table`, so no DDL column-add
  is required for the new key; verify at implementation time that the
  payload contract is documented at the table definition. If the
  payload becomes schema-typed in the future, add the
  `prior_trust_mode` field here.)
- `src/elspeth/web/preferences/routes.py` (Task 1 Sub-task 7a:
  account-level opt-out / opt-in telemetry emit on
  `PATCH /api/composer-preferences`'s `update_preferences` handler;
  depends on the B2 service reshape below)
- `src/elspeth/web/preferences/service.py` (B2: reshape
  `update_composer_preferences` to load the prior record atomically
  and return both prior and current — precondition for the
  account-level opt-out / opt-in telemetry emit. Symmetric to the
  sessions/service.py reshape; see §"Service signature precondition
  (B2 — load-bearing)". **B2.b: NO new audit event** — the
  existing operational-only architecture documented in the
  "Operational signal only" module-level comment in
  `preferences/service.py` is preserved; the account-level
  counters become post-state-only and the route reads only
  `current.default_mode` from the reshape. See §"Account-level
  scope narrowing (B2.b — load-bearing)" for the full rationale
  and the rejected alternative of promoting account-level
  preferences to audit.)
- `src/elspeth/web/sessions/telemetry.py` (Task 2: new counters in container)
- `src/elspeth/web/composer/service.py` (Task 1: wire deferred markers)
- `src/elspeth/web/frontend/src/components/chat/TemplateCards.tsx`,
  `.test.tsx` (Task 3)
- `src/elspeth/web/frontend/src/components/sessions/SessionSidebar.tsx`
  (Task 4: deletion)
- `src/elspeth/web/frontend/src/components/sessions/HeaderSessionSwitcher.tsx`
  (Task 4: filter, archive)
- `src/elspeth/web/frontend/src/App.tsx` (Task 5: regroup shortcuts;
  Task 4: drop sidebar mount)
- `src/elspeth/web/frontend/src/components/common/ShortcutsHelp.tsx`,
  `.test.tsx` (Task 5)
- `src/elspeth/web/frontend/src/components/settings/ComposerPreferencesPanel.tsx`
  (Task 6: mount replay button)
- `src/elspeth/web/frontend/src/api/client.ts` (Task 6 — `updateComposerPreferences` lives here; no separate `api/preferences.ts` module exists; if a future scope expansion wants to split client.ts into per-domain modules, that's a separate decision)
- `src/elspeth/web/frontend/src/stores/preferencesStore.ts` (Task 6)
- `src/elspeth/web/frontend/package.json` (Task 7: jest-axe +
  @types/jest-axe + axe-core dev deps — B7 swap; vitest-axe was
  rejected, see Task 7 Step 1 rationale and the Risks row)
- `src/elspeth/web/frontend/vite.config.ts` (Task 7: register
  `src/test/a11y/setup.ts` in `test.setupFiles` so the
  `toHaveNoViolations` matcher is registered globally before the
  audit suite runs. **Verify exact filename at implementation time**
  — the project may carry a separate `vitest.config.ts`; if so,
  the setupFiles entry belongs there instead. B7.)
- `src/elspeth/web/frontend/src/components/**/*.tsx` (Task 7, 8: a11y
  + polish across new components)
- `src/elspeth/web/shareable_reviews/...` (Task 1 Sub-task 7d — B3
  cohort a: wires `composer.share.token_verify_failure_total` and
  `composer.share.link_expiry_hit_total` at Phase 6's verify-failure
  and expiry-hit branches. Exact file path depends on Phase 6's
  final delivery shape — probe at execution time per Task 0;
  conditional sub-task.)
- `src/elspeth/web/composer/...` or `src/elspeth/web/sessions/...`
  (Task 1 Sub-task 7e — B3 cohort b1: wires
  `composer.interpretation.opt_out_total` at the route that commits
  the `interpretation_source='auto_interpreted_opt_out'` audit row.
  Exact path located at execution time by following the
  audit-source string.)
- `src/elspeth/web/audit_readiness/routes.py` (Task 1
  Sub-task 7f — B3 cohort b2: wires
  `composer.audit.fetch_failure_total` at the audit-readiness
  fetch-failure branch. Exact path depends on Phase 2C's final
  delivery shape — verify at execution time per Task 0.)

**Files this plan deletes:**

```text
src/elspeth/web/frontend/src/components/sessions/SessionSidebar.tsx       (Task 4, conditional)
src/elspeth/web/frontend/src/components/sessions/SessionSidebar.test.tsx  (Task 4, conditional)
```

Note: the SessionSidebar deletion is **conditional** — Task 4 verifies
the HeaderSessionSwitcher exists from Phase 3B before deleting the old
sidebar. If Phase 3B hasn't shipped, Task 4 is a documented no-op.

---

## Verification approach

Each task is TDD-shaped. The order is:

1. Probe (where conditional): does the upstream surface this task
   polishes exist? Branch.
2. Failing test that pins the target behaviour.
3. Implementation.
4. Test passes; existing tests still pass; lint clean; mypy clean.
5. Commit with a message that names the phase and the harvest.

The final task (Task 8) runs the full pytest suite, the full vitest
suite, ruff, mypy, and the freeze-guard enforcer. Phase 8 closes only
when every command exits 0.

### Fixture isolation pinning (Q10 — load-bearing for the Q-cluster)

All counter-observing tests in this phase — Task 1 Step 3's per-helper
tests, Task 2 Step 2's route-level tests including the Q4 combined-PATCH
and Q8 audit-payload contract test, Task 6 Step 7's Q6 transition-edge
tests, and the Q7 conditional emit tests under Sub-tasks 7b / 7d / 7e
/ 7f — **must** use a `function`-scoped `sessions_telemetry` fixture
that rebuilds the container per test via `build_sessions_telemetry()`
(no `meter` argument, so the fake-counter branch is exercised). The
`_FakeCounter` shape used by `observed_value` accumulates its `.calls`
list across the lifetime of the counter instance; a `module`- or
`session`-scoped fixture would share that list across tests, and
`observed_value(...)` assertions would become **order-dependent and
flaky**:

- A test that expects `observed_value(...) == 1` after one
  emission would silently pass at `== 2` if a prior test in the
  same file already emitted to that counter.
- A test that expects `observed_value(...) == 0` (e.g. the
  "no-op write" / "mode unchanged" guards) would fail unpredictably
  depending on which sibling tests ran before it under pytest's
  parallelisation or alphabetical ordering.

The Q1 module-local counter test for
`_PHASE_8_PROBE_FAILED_COUNTER` is an exception to this rule
*because the counter itself is module-local* (constructed at
`telemetry_phase8.py` import time per W8-r2) — that test uses a
fake meter / module-reload pattern rather than the
`sessions_telemetry` fixture, as called out inline at Task 1 Step
3. The Q2 real-meter snapshot test passes its own `fake_meter`
to `build_sessions_telemetry()` and similarly does not consume the
container fixture. Every other counter-observing test in this
phase consumes the function-scoped fixture.

### Probe safety policy

Probes in conditional tasks check whether an upstream phase shipped
its surface. **If a probe returns "not found":**

- The conditional task is a documented no-op (record in
  `21-phase-9-followups.md`).
- The metric is **not** emitted with degraded/inferred values.
  Partial telemetry is worse than no telemetry — fail safe.
- Emit `composer.phase_8.probe_failed_total{phase=X, probe=Y}` via
  the OTel meter so the absence is visible in dashboards. This
  counter signals "this conditional task could not run," not an
  error. Wired as a module-local counter in `telemetry_phase8.py`
  (`_PHASE_8_PROBE_FAILED_COUNTER`), not via the `_SessionsTelemetry`
  container — see W8-r2 module-local counter resolution.

### Phase 9 follow-ups file — review cadence (S4 — load-bearing)

The `docs/composer/ux-redesign-2026-05/21-phase-9-followups.md` file
is created lazily by **five** sites in this plan: Task 0 Step 4
(B3 cohort (c) seeding); Task 0 Step 2 (probe-failure records);
Task 1 Sub-tasks 7b/7c/7d/7e/7f Case-B branches; Task 4 Case B-1;
Task 6 Case B-2; Task 7 Step 5/7 (medium/low a11y findings). Without
an explicit review cadence the file will accumulate entries and rot
unread — the same anti-pattern the project-memory entry
`project_cicd_allowlist_audit_2026-05-19` documents for the
tier-model allowlist (51% growth in 32 days because no expiry
mechanism forced periodic review).

To prevent the same rot here:

- **TTL in the file header.** When Task 0 Step 4 (the canonical
  first-creation site) seeds the file, the header MUST include a
  TTL paragraph: "Review by [date = Phase 8 close + 90 days]; if
  Phase 9 has not begun by that date, each remaining entry must
  either be filed as a Filigree issue, promoted into a Phase 9
  plan section, or explicitly dismissed with a one-line rationale.
  Letting the file age past TTL without review breaks the deferral
  contract."
- **Phase 8 close gate.** Task 8 Step 7 adds a check:
  `wc -l docs/composer/ux-redesign-2026-05/21-phase-9-followups.md`.
  If the result is **>30 lines** at Phase 8 close, surface to the
  operator: "Phase 9 follow-ups file has grown beyond the soft cap;
  triage before phase close or accept that Phase 9 ingest will be
  expensive." 30 lines is roughly 10–15 entries (each entry is 1–3
  lines); above that, the file is too dense for a Phase 9 scoping
  read to be cheap.
- **Phase 9 ingest contract.** When Phase 9 begins, its Task 0
  ingests every entry in `21-phase-9-followups.md` and either:
  promotes the entry into a Phase 9 task; files a Filigree issue
  with a back-link; or explicitly dismisses the entry with a
  one-line rationale committed back into the file. Un-ingested
  entries are surfaced to the operator before Phase 9 close.

Cross-reference: this cadence policy is named in Task 0 Step 4 (so
the implementer seeding the file knows to include the TTL header)
and in Task 8 Step 7 (so the implementer closing the phase runs
the line-count check).

---

## Task 0: Probe-mechanism tests

**Why first:** verifies that the probe commands each conditional task
relies on actually return the expected signal on the live codebase
before any conditional branch runs. This is a lightweight manual
check, not a probe-framework extraction — probes remain inline bash
one-liners; this task asserts each returns a known value.

- [ ] **Step 1: Run each conditional probe and record the result**

For each conditional probe below, run the command and record
whether it returned a hit or a miss:

| Task | Probe target | Expected if phase shipped |
|------|--------------|--------------------------|
| Task 1 | `grep -rn 'telemetry: deferred to Phase 8'` | ≥1 match |
| Task 1 Sub-task 7d (B3 cohort a) | `grep -rn 'verify_token\|verify_share_token' src/elspeth/web/` | ≥1 match (Phase 6 token-verify path) |
| Task 1 Sub-task 7e (B3 cohort b1) | `grep -rn 'auto_interpreted_opt_out' src/elspeth/web/` | ≥1 match (Phase 5b opt-out fact) |
| Task 1 Sub-task 7f (B3 cohort b2) | `grep -rn 'audit_readiness\|composer/audit/readiness' src/elspeth/web/` | ≥1 match (Phase 2C audit-readiness endpoint) |
| Task 4 | `ls ...HeaderSessionSwitcher.tsx` | file exists |
| Task 6 | `grep -rn 'tutorial_completed_at'` in web/ | ≥1 match |

- [ ] **Step 2: For each missed probe, emit the probe-failure counter**

The probe-failure counter is module-local in `telemetry_phase8.py`
(see Task 1 Step 4 module shape), constructed at import time via
`meter.create_counter`. No container slot is required, so Task 0 can
run in any order relative to Task 1 Step 5. The W8-r2 module-local
counter resolution dissolves the prior pass-1 bootstrap-order
coupling that required Task 1 Step 5 to land first; both tasks are
now structurally self-contained with respect to this counter.

```python
# In telemetry_phase8.py module scope (declared at import time):
from elspeth.web.composer.telemetry_phase8 import _PHASE_8_PROBE_FAILED_COUNTER
_PHASE_8_PROBE_FAILED_COUNTER.add(1, attributes={"phase": "X", "probe": "Y"})
```

Record in `21-phase-9-followups.md` for each miss.

- [ ] **Step 3: Confirm no conditional task proceeds on a missed probe**

Review each task whose probe returned a miss. Confirm the plan
routes it to Case B-1/B-2 (documented no-op or operator surface).
No task may emit a metric whose upstream signal was never recorded.

- [ ] **Step 4: Seed `21-phase-9-followups.md` with B3 cohort (c) entries + Decision 2 deferral**

Per §"Cross-phase telemetry — cohort split (B3 reshape)" → "Cohort
(c) — Phase 9 follow-up filing", Phase 8 explicitly declines to
backfill two perf-only metrics into shipped phases. Per the
Decision 2 resolution (Option C — see §"Decision 2" above),
Phase 8 also defers the `composer.tutorial.replayed_total` counter
and the audit-vs-telemetry boundary question it raises. File all
three now so the deferrals are durable and discoverable:

- If `docs/composer/ux-redesign-2026-05/21-phase-9-followups.md`
  does not yet exist, create it with a short header explaining its
  role (a running list of Phase 9 follow-ups accumulated during
  Phase 8 close). The header MUST include the TTL paragraph
  mandated by §"Phase 9 follow-ups file — review cadence (S4 —
  load-bearing)" — set the TTL to "Phase 8 close + 90 days" using
  the actual close date once known (placeholder `[TBD: Phase 8
  close + 90d]` is acceptable at Task 0 time and is updated by Task
  8 Step 7 when Phase 8 actually closes).
- Append three entries:

  - `composer.audit.render_duration` — owner Phase 2C (shipped);
    proposed emit site: audit-panel render path. Rationale:
    perf-only, no security impact; Phase 8 declined to reopen a
    shipped phase for perf instrumentation.
  - `composer.interpretation.resolve_duration` — owner Phase 5b
    (shipped); proposed emit site: interpretation resolve path.
    Rationale: same as above.
  - `composer.tutorial.replayed_total` + **boundary question** —
    owner Phase 9; proposed emit site: the `prior.tutorial_completed_at
    is not None and current.tutorial_completed_at is None` transition
    gate in `update_composer_preferences` that Phase 8 Task 6 Step 7
    would have implemented (the retake PATCH body is
    `{"tutorial_completed_at": null}` per the Phase 4 cross-plan
    contract). **Open question Phase 9 MUST answer first:** is a
    deliberate tutorial-replay click (a user-write-intent) a
    "non-decision" under the CLAUDE.md superset exception, or does
    it merit a Landscape audit row? Phase 8 reviewed this and ruled
    the superset exception applies to read-path operational health
    only; broadening it would be a project-wide policy change.
    Phase 9 may either (a) accept that ruling and ship the counter
    as telemetry-only with a matching audit row for the transition
    (capturing the destroyed prior timestamp), (b) propose an
    audit-vocabulary amendment that broadens the superset exception
    project-wide and ship the counter telemetry-only, or (c) drop
    the counter entirely if downstream consumers turn out not to
    need it. The three Q6 transition-edge tests
    (`set_to_null` [retake], `null_to_set` [completion], `no_op`)
    that Phase 8 spec'd are re-usable when Phase 9 resumes the work
    — note that they originally targeted a `bool` field; the
    transition predicates change to `datetime|None` shape.

This is a documentation-only step; no code changes.

---

## Task 1: Wire telemetry-deferred markers seeded by earlier phases

**Files:**

- Search target: every file under `src/elspeth/web/` and
  `src/elspeth/web/frontend/src/` for the marker
  `# telemetry: deferred to Phase 8` (Python) or
  `// telemetry: deferred to Phase 8` (TS).
- Modify: each file containing a marker, to replace it with a real
  counter emit.
- Modify: `src/elspeth/web/sessions/telemetry.py` — add any new
  counter names to the `_SessionsTelemetry` dataclass and the
  `build_sessions_telemetry()` factory.
- Create: `src/elspeth/web/composer/telemetry_phase8.py` — module
  that aggregates the Phase 8 emit helpers (a thin wrapper around the
  counter container so call-sites don't import the dataclass
  directly).
- Create: `src/elspeth/web/composer/telemetry_phase8_test.py`.

**Why first:** establishes emit conventions that Tasks 2 and 6 reuse.

### Bootstrap step (A9 — tracer-bullet first)

- [ ] **Step 0: Land ONE counter end-to-end before scaling (A9)**

Pass-2 review caught that this task scales to **12 new counters**
extended into the container, **3 audit-payload + service-signature
preconditions** (B1, B2, B2.b), and **6 emit sub-tasks** (7a–7f).
Landing all of that in one mechanical pass amplifies integration
risk: a mis-wired bootstrap-order issue, an OTel exporter-config
quirk, a fixture-rebuild defect, or a B1/B2 ordering mistake would
all manifest as an opaque cluster of failures after the full
container extension landed.

The tracer-bullet discipline: pick the **smallest** counter that
exercises the full pipeline end-to-end, and land it through every
step (B1 audit-payload precondition co-land → B2 service-signature
reshape → container slot → helper → route emit → Q-cluster test →
manual staging verification) **before** extending any other counter.

Recommended tracer: **`composer.mode.opted_out_total`** (account-
level opt-out). Rationale:

- Post-state-only per §B2.b — no transition predicate to get wrong;
  no `from_mode` attribute to drift; the helper takes no kwargs.
- Lives in `preferences/routes.py` whose existing
  `composer.preferences.patch_total` emit shows the working
  reference pattern (the `_PREFERENCES_PATCH_COUNTER` site at
  `preferences/service.py:85` is the file's existing OTel counter
  precedent).
- Does **not** depend on the B1 audit-payload extension (B2.b
  vacuously satisfies the superset rule — no transition data in
  the counter, no transition data required in audit). The B1
  co-land discipline is exercised by the B2 service-signature
  reshape on `preferences/service.py`, which still happens for
  code-shape symmetry per §B2 even though `prior.default_mode` is
  unused at this tracer's emit site.
- One Q-cluster test (Q1-shaped: the per-helper increment
  assertion) is enough to gate the tracer; the broader Q-cluster
  remains as scaffolded for the post-tracer pass.

**Tracer landing sequence (one PR per box, OR all in one cohesive PR
labelled "Phase 8 tracer"):**

1. B2 reshape `preferences_service.update_composer_preferences`
   (atomic read-prior-record + write + return both) + Phase 1A
   fixture co-land. No emit yet. Test suite green.
2. Add the `mode_opted_out_total` slot to `_SessionsTelemetry`
   (Step 5's container extension, scoped to this one field), plus
   the `record_mode_opted_out(tel)` helper (Step 4's module shape,
   scoped to this one helper) with the W5 try/except wrap. Add
   the Q1-shaped per-helper test from Step 3 (scoped to this one
   helper). Test suite green.
3. Wire the emit at `update_preferences` in
   `preferences/routes.py` per Sub-task 7a (scoped to the
   `default_mode=freeform` branch only). Add the route-level test
   that the counter increments on a `freeform` PATCH. Test suite
   green.
4. Deploy to staging per `project_staging_deployment.md` memory
   (`npm run build`; `systemctl restart elspeth-web.service`).
   **Precondition:** the 8a MeterProvider wiring per §"MeterProvider
   precondition (B1-r3 — load-bearing)" must already be on the
   deployed binary; if it isn't, this step's validation cannot
   succeed and the tracer is mis-sequenced. Hit the PATCH endpoint
   manually with body `{"default_mode": "freeform"}`. Then validate
   the counter is observable via:

   ```bash
   curl -s https://elspeth.foundryside.dev/metrics \
     | grep -E '^composer_mode_opted_out_total\b'
   ```

   Expected output: at least one line of the form
   `composer_mode_opted_out_total{...} <N>` where `N >= 1`. (The
   Prometheus exporter translates `.` → `_` in metric names per
   OTel semantic conventions.) If the grep returns nothing, do
   **not** proceed to Step 5 of the wider Task 1: either the
   counter is mis-named, the provider is not wired, the helper
   never ran, or the W5 try/except swallowed a real error. Each of
   those is a tracer failure to diagnose before any further counter
   is added. Record the curl output and the timestamp in the
   tracer-bullet validation note for the commit.

**Gate:** only after the tracer is green end-to-end (steps 1–4
above) do Steps 4–7 below extend with the remaining 11 counters
and 5 sub-tasks. If the tracer reveals an integration defect, fix
it with one counter on the wire — cheap to iterate — before
amplifying the problem across the full container.

The tracer-bullet pass is **additive scaffolding** for the existing
TDD steps, not a replacement: Steps 1–7 below still run, but they
start from a known-good tracer rather than from cold. The Risks
table calls this out as a deliberate Phase 8 discipline; pass-2
review noted that without it, the plan was structurally one large
unit of risk rather than a sequence of cheap-to-revert units.

### Probe step

- [ ] **Step 1: Probe — enumerate all deferred markers**

Run:

```bash
grep -rn 'telemetry: deferred to Phase 8\|TODO.*phase.?8.*telemetry\|TODO.*telemetry.*phase.?8' \
  --include='*.py' --include='*.ts' --include='*.tsx' \
  /home/john/elspeth/src/elspeth/
```

Expected outcomes:

- **Case A — markers found:** record each location. Each becomes a
  numbered sub-task below (Step 4a/4b/4c/…).
- **Case B — no markers found:** the earlier phases either inlined
  the telemetry on land, or used a different marker convention.
  Re-run with broader patterns:

```bash
grep -rn '# telemetry:\|// telemetry:\|TODO.*telemetry\|FIXME.*telemetry' \
  --include='*.py' --include='*.ts' --include='*.tsx' \
  /home/john/elspeth/src/elspeth/ | grep -iv 'test\|spec' | head -40
```

- **Case C — still no markers, but design doc 10 names three:**
  the Phase 1A PATCH route, the Phase 5a dynamic-source emit, and
  the Phase 6 YAML-export event. Find each by call-site:

```bash
grep -n 'update_composer_preferences\|composer_preferences' \
  /home/john/elspeth/src/elspeth/web/sessions/routes.py | head -10
```

The PATCH route MUST have a counter emit; the Phase 1A plan says it
will. If the emit is missing, that's the marker — add it as Step 4a.

### TDD steps

- [ ] **Step 2: Inspect the existing telemetry container**

Read `/home/john/elspeth/src/elspeth/web/sessions/telemetry.py` lines
119-168. Note the `_SessionsTelemetry` dataclass shape and the
`build_sessions_telemetry()` factory. The container is `frozen=True
slots=True`; new counters are added by extending both the dataclass
**and** the factory body.

The convention for counter naming, from the existing container:

- `composer.audit.<subject>_total` for audit-related counters.
- `composer.<subject>_total` for non-audit operational counters.
- All counters end in `_total` (OTel cumulative-counter convention).

This phase's new counters live under `composer.mode.*` (Task 2),
`composer.session.switched_total` (Task 2), `composer.tutorial.*`
(Task 6 conditional).

- [ ] **Step 3: Write the failing test — `telemetry_phase8_test.py`**

Create `/home/john/elspeth/src/elspeth/web/composer/telemetry_phase8_test.py`.
The file uses `build_sessions_telemetry()` (no meter → fake counters)
plus the `observed_value` helper from `sessions/telemetry.py`, and
asserts the helpers in `telemetry_phase8.py` produce the canonical
counter increments. Required test cases (names indicative):

- `test_record_mode_opted_out_increments_counter` —
  `record_mode_opted_out(tel)`; assert
  `observed_value(tel.mode_opted_out_total) == 1`; assert the
  recorded attribute dict is exactly `{}` (read via the `calls`
  attribute after type-narrowing to `_FakeCounter`). The helper
  takes no `from_mode` kwarg per §"Account-level scope narrowing
  (B2.b — load-bearing)".
- `test_record_mode_opted_in_increments_counter` — symmetric on
  `tel.mode_opted_in_total`; empty attribute dict.
- `test_record_session_switched_increments_counter` —
  `record_session_switched(tel, from_mode="explicit_approve",
  to_mode="auto_commit")`; assert
  `observed_value(tel.session_switched_total) == 1`; assert the
  recorded attribute dict is exactly `{"from_mode":
  "explicit_approve", "to_mode": "auto_commit"}`. This helper is
  per-session and retains its transition-shaped attributes; the B1
  precondition on the per-session `trust_mode.changed` audit event
  keeps the superset rule satisfied here. The values are drawn from
  the `trust_mode` CHECK constraint in
  `src/elspeth/web/sessions/models.py:150`, **not** from the
  account-level `default_composer_mode` vocabulary (see §"Vocabulary
  discipline (B1-r2 — load-bearing)").
- Parametrise the `from_mode` and `to_mode` arguments of
  `record_session_switched` over `{"explicit_approve",
  "auto_commit"}` and assert each combination is accepted. Do
  **not** parametrise over `"guided"`, `"freeform"`, or
  `"unknown"`; those are wrong vocabulary (account-level) or
  fabricated values that the per-session column does not admit.
- `test_record_session_switched_rejects_invalid_mode` — call with
  `from_mode="guided"` and assert `ValueError` matching
  `"from_mode must be"`; call again with `from_mode="yolo"` and
  assert the same. The `"guided"` case is the canonical
  cross-vocabulary regression assertion for B1-r2: `"guided"` is
  *valid* for the account-level `default_composer_mode` column and
  *invalid* for the per-session `trust_mode` column, and the pass-1
  draft would have admitted it through a shared Literal. The
  `"yolo"` case is the generic out-of-set guard.
- The opt-out / opt-in helpers take no mode argument; there is no
  ValueError to assert against them. Do **not** add a
  parametrised-invalid-mode test for those helpers.
- **(Q1 — probe-failed increment test, module-local counter):**
  `test_phase_8_probe_failed_counter_records_phase_and_probe_attributes` —
  assert that calling `_PHASE_8_PROBE_FAILED_COUNTER.add(1,
  attributes={"phase": "Task 4", "probe": "HeaderSessionSwitcher"})`
  records exactly one observation with both attribute keys present
  and the expected values. **Test-pattern note:** unlike the
  surrounding container-based tests, this counter is a module-local
  `metrics.get_meter()` counter constructed at `telemetry_phase8.py`
  import time (per W8-r2 — see Task 1 Step 4 module shape) and is
  **NOT** a slot on `_SessionsTelemetry`, so the `_FakeCounter`
  `.calls`-list / `observed_value` pattern used by every other test
  in this file does not apply. Instead, mock `metrics.get_meter()`
  (or substitute a fake meter object that captures
  `create_counter().add()` calls) before module import, then either
  reload the module or import it freshly inside the test so the
  module-level `_meter = metrics.get_meter(__name__)` line picks up
  the fake. Assert on the fake's recorded calls. The contrast with
  the container-based tests is deliberate — one sentence of prose
  in the test docstring explaining why the pattern differs prevents
  a future maintainer from "harmonising" this test into the wrong
  shape.

- **(Q2 — real-meter snapshot test):**
  `test_factory_registers_canonical_counter_names` — call
  `build_sessions_telemetry(meter=fake_meter)` where `fake_meter`
  is a stub object whose `create_counter(name=..., description=...)`
  records the `name=` argument of every invocation into a list.
  Assert the recorded names match the expected set **exactly** as a
  hard-coded snapshot (no superset / subset assertion — exact
  equality, so a missing OR extra name fails the test). Enumerate
  all twelve wire-name strings explicitly:

  ```python
  expected_names = [
      "composer.mode.opted_out_total",
      "composer.mode.opted_in_total",
      "composer.session.switched_total",
      "composer.tutorial.started_total",
      "composer.tutorial.completed_total",
      # composer.tutorial.replayed_total — deferred to Phase 9
      # per Decision 2 resolution (Option C). Do NOT add it back
      # to expected_names without re-opening that decision.
      "composer.session.completed_total",
      "composer.share.token_verify_failure_total",
      "composer.share.link_expiry_hit_total",
      "composer.interpretation.opt_out_total",
      "composer.audit.fetch_failure_total",
      "composer.source.dynamic_created_total",
  ]
  ```

  Rationale: this is the **only** test that catches a typo in the
  real-meter branch of `build_sessions_telemetry()` (e.g.
  `composer.share.token_verify_failure_totals` with a trailing `s`,
  or `composer.mode.opt_out_total` losing the `ed`). Per-helper
  tests above exercise the fake-counter branch; the real-meter
  branch is wired by string literal and has no other coverage. The
  exact-equality assertion also catches accidental *additions* —
  if a future PR adds a counter without updating this snapshot,
  the test fails until the snapshot is intentionally updated,
  forcing the new wire name through code review.

  Note: `composer.phase_8.probe_failed_total` is **not** in this
  list. Per W8-r2 it is a module-local counter on
  `telemetry_phase8.py`, not a `_SessionsTelemetry` slot, and is
  covered by Q1's test above.

Run: `.venv/bin/python -m pytest src/elspeth/web/composer/telemetry_phase8_test.py`
Expected: import error (telemetry_phase8.py does not exist yet).

- [ ] **Step 4: Implement `telemetry_phase8.py`**

Create `/home/john/elspeth/src/elspeth/web/composer/telemetry_phase8.py`.
Module docstring: "Phase 8 emit helpers for composer mode and tutorial
telemetry. Wraps the `_SessionsTelemetry` counter container; helpers
are pure functions and call-sites pass the container in. Composer
imports this module; sessions code must not (ownership-vs-metric
rule per `sessions/telemetry.py`). Trust tier: Tier 2 throughout;
inputs are Literal-typed; runtime guards are offensive."

Module shape:

- **Module-level (W8-r2 module-local counter):**
  ```python
  _meter = metrics.get_meter(__name__)
  _PHASE_8_PROBE_FAILED_COUNTER = _meter.create_counter(
      name="composer.phase_8.probe_failed_total",
      description="Phase 8 conditional probe missed an upstream phase surface",
  )
  ```
  Declared at module import time. Matches the existing
  `_PREFERENCES_PATCH_COUNTER` pattern in
  `src/elspeth/web/preferences/service.py:85` (acquire the meter via
  `metrics.get_meter(__name__)`; pick the exact call site to mirror
  that file). Used by Task 0 Step 2's probe-failure emit; it is
  **NOT** a slot on `_SessionsTelemetry`. This avoids the frozen +
  `slots=True` bootstrap-order coupling that pass-1 introduced (per
  W8-r2 / A5): had the counter been a container field, Task 0 Step
  2's emit would have required Task 1 Step 5 (container extension)
  to land first, or AttributeError on the slots dataclass. The
  module-local placement makes Task 0 and Task 1 structurally
  independent with respect to this counter. Naming: the OTel name
  ends in `_total` per project convention; the Python constant name
  uses SCREAMING_SNAKE_CASE per the `_PREFERENCES_PATCH_COUNTER`
  precedent.
- Re-export the container as `SessionsTelemetry` (avoid leaking the
  underscore name across the composer surface).
- `_SessionTrustMode = Literal["explicit_approve", "auto_commit"]`
  and a matching frozenset `_KNOWN_SESSION_TRUST_MODES` for the
  runtime check. The literal set matches the CHECK constraint on
  `src/elspeth/web/sessions/models.py:150`'s `trust_mode` column
  exactly. This type is used **only** by `record_session_switched`
  (per-session, B1-extended audit). The account-level helpers
  (`record_mode_opted_out` / `record_mode_opted_in`) are
  post-state-only per §"Account-level scope narrowing (B2.b —
  load-bearing)" and take no mode kwarg, so they need no Literal.
- **Deliberately absent:** no `_ModeName = Literal["guided",
  "freeform", "unknown"]` and no shared `_KNOWN_MODES`. A pass-1
  draft defined a single mode Literal shared across the account-
  level and per-session helpers; the pass-2 review (B1-r2) caught
  that the per-session helper's runtime inputs come from the
  `trust_mode` column whose CHECK constraint admits only
  `'explicit_approve'` / `'auto_commit'` (per `models.py:150`), so
  every per-session emit would assert-fail. The fabricated
  `'unknown'` value was a parallel defect — neither `trust_mode`
  nor `default_composer_mode` (per `models.py:1076`) admits NULL
  or `'unknown'`. See §"Vocabulary discipline (B1-r2 —
  load-bearing)" for the full root-cause narrative.
- `_assert_session_trust_mode(name, value)` — raises `ValueError`
  with message `f"{name} must be one of
  {sorted(_KNOWN_SESSION_TRUST_MODES)!r}; got {value!r}"` when
  value is outside the literal set. Used by
  `record_session_switched` only.
- `record_mode_opted_out(tel)` — calls
  `tel.mode_opted_out_total.add(1, attributes={})` with no kwargs and
  an empty attribute dict. Account-level, post-state-only per
  §"Account-level scope narrowing (B2.b — load-bearing)". No
  mode-validation needed (no mode kwarg).
- `record_mode_opted_in(tel)` — symmetric on
  `tel.mode_opted_in_total`. Also kwarg-free, attribute-free, and
  validation-free.
- `record_session_switched(tel, *, from_mode, to_mode)` —
  annotations `from_mode: _SessionTrustMode, to_mode:
  _SessionTrustMode`. Asserts both via
  `_assert_session_trust_mode`, then
  `tel.session_switched_total.add(1, attributes={"from_mode": ...,
  "to_mode": ...})`. Per-session; retains the transition shape
  because the per-session `trust_mode.changed` audit event carries
  both prior and new state under the B1 extension. Vocabulary is
  `'explicit_approve'` / `'auto_commit'` (per `models.py:150`),
  **not** `'guided'` / `'freeform'` — see §"Vocabulary discipline
  (B1-r2 — load-bearing)".

**OTel exporter failure handling (W5 — load-bearing).** Every
`record_*` helper above MUST wrap the underlying
`tel.<counter>.add(...)` call in a `try` / `except Exception` block
that swallows the exception and returns `None`. Rationale: per
CLAUDE.md "Telemetry and Logging", telemetry is best-effort; a
broken OTel exporter must not 500 a PATCH that has already written
its audit row (B1 ordering: audit fires sync + crash-on-failure
*before* the emit; if the emit then raises, the audit record stands
but the user request fails after the fact, which inverts primacy).
Concretely:

```python
def record_mode_opted_out(tel: SessionsTelemetry) -> None:
    try:
        tel.mode_opted_out_total.add(1, attributes={})
    except Exception:
        # Telemetry-only exemption per CLAUDE.md
        # logging-telemetry-policy: counter-emit failures are
        # silently swallowed; the audit row already wrote.
        return None
```

The `_assert_session_trust_mode` ValueError is the **only**
exception that escapes (it's a programmer-error guard, not an
OTel-exporter failure). Wrap **after** the assert so input
validation still crashes loudly. The assert-then-swallow ordering
is exactly the surface where B1-r2 manifested — a cross-vocabulary
input (e.g. `from_mode="guided"` when the per-session column wants
`"explicit_approve"`) raises before the swallow can catch it, and
the user PATCH 500s after the audit row stands. The fix is at the
Literal definition (see §"Vocabulary discipline (B1-r2 —
load-bearing)" and the helper-module shape above); the
assert-before-swallow ordering itself is correct and must stay.

```python
def record_session_switched(
    tel: SessionsTelemetry,
    *,
    from_mode: _SessionTrustMode,
    to_mode: _SessionTrustMode,
) -> None:
    _assert_session_trust_mode("from_mode", from_mode)
    _assert_session_trust_mode("to_mode", to_mode)
    try:
        tel.session_switched_total.add(
            1, attributes={"from_mode": from_mode, "to_mode": to_mode}
        )
    except Exception:
        return None
```

**Test the swallow behaviour.** Add to Task 1 Step 3's failing-test
enumeration: a test that constructs a `SessionsTelemetry` with a
counter whose `add()` raises `RuntimeError("simulated exporter
failure")`, calls each `record_*` helper against it, and asserts
the helper returns `None` without propagating the exception. The
fake-counter pattern from `sessions/telemetry.py`'s test surface
already supports this — extend the `_FakeCounter` shape with an
optional `raise_on_add` flag, or use a `MagicMock` whose `add`
side-effect raises.

- [ ] **Step 5: Extend the counter container**

Modify `/home/john/elspeth/src/elspeth/web/sessions/telemetry.py` —
extend `_SessionsTelemetry` and `build_sessions_telemetry()` with the
new counters. **Do not rename or reorder existing fields** (the
dataclass is `frozen slots=True` and consumers depend on field
order via dataclass introspection).

Add at the end of the existing field list:

```python
mode_opted_out_total: _Counter
mode_opted_in_total: _Counter
session_switched_total: _Counter
# Tutorial counters wired by Task 6, conditional. They are added
# unconditionally to the container so the helper module compiles;
# unused counters cost effectively nothing.
tutorial_started_total: _Counter
tutorial_completed_total: _Counter
# tutorial_replayed_total — deferred to Phase 9 per Decision 2
# resolution (Option C). The replay button ships without the
# counter slot; Phase 9 adds both the slot and the boundary-
# question resolution (audit-row vs telemetry-only).
session_completed_total: _Counter
# B3 cohort (a) — Phase 6 share-counter emits (Sub-task 7d).
# Added unconditionally; remain unused if Phase 6 token-verify path
# has not shipped at execution time (probe in Task 0 decides).
share_token_verify_failure_total: _Counter
share_link_expiry_hit_total: _Counter
# B3 cohort (b1) — Phase 5b interpretation opt-out (Sub-task 7e).
interpretation_opt_out_total: _Counter
# B3 cohort (b2) — Phase 2C audit-readiness fetch failure (Sub-task 7f).
# Telemetry-only signal; superset exception for non-decision read.
audit_fetch_failure_total: _Counter
# B4 (W8-r2 module-local counter) — DELIBERATELY ABSENT: the Task 0
# probe-failure counter is NOT a field on this container. Per W8-r2 /
# A5, it lives as a module-local OTel counter in telemetry_phase8.py
# (`_PHASE_8_PROBE_FAILED_COUNTER`), constructed at import time via
# `meter.create_counter`. This matches the existing
# `_PREFERENCES_PATCH_COUNTER` pattern in
# `src/elspeth/web/preferences/service.py:85`. The motivation is to
# remove the bootstrap-order coupling that would otherwise require
# Task 1 Step 5 to land before Task 0 Step 2 could emit. See Task 1
# Step 4 module shape + §Risks "Phase 8 probe-failure counter
# bootstrap-order coupling (W8-r2)".
# B5 — Phase 5a dynamic-source emit (Sub-task 7b). Added unconditionally
# so the conditional emit can compile; emit fires only if Phase 5a's
# dynamic-source-from-chat path has shipped (probe in Task 0).
source_dynamic_created_total: _Counter
```

And in `build_sessions_telemetry()` extend both branches (`if meter
is None` and the real-meter branch). Counter names in the
real-meter branch:

- `composer.mode.opted_out_total`
- `composer.mode.opted_in_total`
- `composer.session.switched_total`
- `composer.tutorial.started_total`
- `composer.tutorial.completed_total`
- ~~`composer.tutorial.replayed_total`~~ — deferred to Phase 9
  (Decision 2 / Option C); see `21-phase-9-followups.md`.
- `composer.session.completed_total`
- `composer.share.token_verify_failure_total` (B3 cohort a)
- `composer.share.link_expiry_hit_total` (B3 cohort a)
- `composer.interpretation.opt_out_total` (B3 cohort b1)
- `composer.audit.fetch_failure_total` (B3 cohort b2)
- `composer.source.dynamic_created_total` (B5 — Phase 5a dynamic-source-from-chat emit; conditional)

Note: `composer.phase_8.probe_failed_total` is **not** wired into
this container per W8-r2 module-local counter resolution. It is
constructed at module-import time inside `telemetry_phase8.py` as
`_PHASE_8_PROBE_FAILED_COUNTER` (see Task 1 Step 4 module shape) and
emitted directly by Task 0 Step 2 without a container slot.

- [ ] **Step 6: Run the test**

`.venv/bin/python -m pytest src/elspeth/web/composer/telemetry_phase8_test.py -v` — all tests pass.

- [ ] **Step 7: Wire deferred markers (per Step 1's enumeration)**

For each marker location found in Step 1, replace the marker comment
with the appropriate emit call from `telemetry_phase8.py`.

**Sub-task 7a (always required — account-level default-mode PATCH route):**

This sub-task wires the **account-level** opt-out / opt-in emit only:
the user's persisted default-mode preference flipping between
`guided` and `freeform`. The per-session `trust_mode` switch is a
different route in a different module and belongs to Task 2 (see
Sub-task 7a' below for the explicit handoff).

**Scope note (B2.b — load-bearing).** Per §"Account-level scope
narrowing (B2.b — load-bearing)" this account-level emit does
**not** require a transition-shaped audit event. The
`preferences_service` deliberately writes no Landscape row
(see the "Operational signal only" module-level comment in
`preferences/service.py`); Phase 8 preserves that
architectural decision and shapes the counter as a **post-state
counter** with no `from_mode` attribute. The B1 audit-payload
precondition (record both prior and new state) therefore applies
**only** to the per-session emit (Task 2 Step 3) and does **not**
apply to this sub-task. There is no symmetric B1 obligation here.

**Precondition (B2):** see §"Service signature precondition (B2 —
load-bearing)" — the account-level
`preferences_service.update_composer_preferences` still reshapes
under B2 (the reshape is symmetric across both service functions
for code-shape consistency, even though only one of them feeds a
transition-shaped counter). The reshape gives the route handler
access to both prior and current; this sub-task uses only
`current.default_mode` and ignores `prior.default_mode`. A
future phase that promotes account-level preferences to audit
(see §B2.b rejected option) would then start consuming
`prior.default_mode` without re-reshaping the service.

In `update_preferences` in `src/elspeth/web/preferences/routes.py`
(the `update_preferences` route handler, mounted at
`PATCH /api/composer-preferences`), after the existing operational
counter (`composer.preferences.patch_total`) increments and before
the response is constructed, branch on **the new state only**:
whenever the PATCH body sets `default_mode=freeform`, call
`record_mode_opted_out(telemetry)`; whenever the PATCH body sets
`default_mode=guided`, call `record_mode_opted_in(telemetry)`.
Neither helper takes a `from_mode` argument; both fire
unconditionally on the post-state regardless of whether the
mode actually changed (the design-doc-10 "opt-out rate" is a
ratio over total PATCHes, not a transition-conditional count).
Audit primacy: there is no audit event for this preference write
(per the architectural decision cited above); the counter is the
sole operational signal and the existing `patch_total` counter
remains the denominator.

**Sub-task 7a' (handoff — per-session session-switched emit):**

The per-session `trust_mode` switch — i.e., a user changing the
mode for a single session via
`PATCH /api/sessions/{session_id}/composer/preferences` (handler
`update_composer_preferences` in
`src/elspeth/web/sessions/routes.py`) — is the **session-switched**
signal and is wired in Task 2 Step 3. It is intentionally not
emitted from Sub-task 7a: the two events live in different route
files, call different services, and write to different audit tables.
Each emit belongs to exactly one route file; this sub-task names the
boundary so the synthesizer can verify that the account-level and
per-session signals are not conflated.

**Sub-task 7b (conditional — Phase 5a dynamic-source emit):**

Probe: `grep -n 'dynamic.source\|dynamic_source' src/elspeth/web/composer/service.py | head -10`

If the dynamic-source-from-chat path exists from Phase 5a but the
emit is missing, add a new counter `composer.source.dynamic_created_total`
to the container (Step 5) and emit it at the dynamic-source creation
site. If the path doesn't exist (Phase 5a didn't ship), record
this as a Phase 9 follow-up and continue.

**(Q7 — conditional emit test for Sub-task 7b):** add a route-level
test that mirrors the probe gate above and asserts the counter
increments on a dynamic-source creation call. The test must be
**skipped** if the Phase 5a route is absent (so the test suite stays
green when Phase 5a hasn't shipped), and **active + asserting** when
the route exists. Suggested skeleton:

```python
import pytest

# Probe target: the Phase 5a dynamic-source-from-chat route symbol.
# Captured at import time so the skipif decorator evaluates once.
try:
    from elspeth.web.composer.service import create_dynamic_source_from_chat  # noqa: F401
    _PHASE_5A_SHIPPED = True
except ImportError:
    _PHASE_5A_SHIPPED = False


@pytest.mark.skipif(
    not _PHASE_5A_SHIPPED,
    reason="Phase 5a dynamic-source-from-chat path not shipped; "
           "Sub-task 7b is a documented no-op per Task 0 probe.",
)
def test_dynamic_source_emits_source_dynamic_created_total(
    sessions_app_client, sessions_telemetry,
):
    # Trigger the dynamic-source creation path via its route or
    # service entry point (exact call shape to be filled in at
    # implementation time once Phase 5a's surface is known).
    ...
    assert observed_value(
        sessions_telemetry.source_dynamic_created_total
    ) == 1
```

Verify the `from elspeth.web.composer.service import …` symbol name
against Phase 5a's actual export at implementation time; if the
symbol name differs, update the `try` block to import the real
symbol. The `_PHASE_5A_SHIPPED` constant is the test-file analogue
of Task 0 Step 1's probe — both must agree, and if Task 0's probe
hits while this constant is False, the symbol name has drifted and
the test needs updating, not skipping.

**Sub-task 7c (conditional — Phase 6 YAML-export event):**

Probe: `grep -rn 'export_yaml\|Export YAML' src/elspeth/web/frontend/src/ | head -10`

If a YAML-export completion verb exists (Phase 6 landed), wire
`composer.session.completed_total{completion_verb="export_yaml"}`
at the click handler's network-success branch. If Phase 6 didn't
ship, the counter exists in the container (added in Step 5) but
nothing emits to it; record as Phase 9 follow-up.

**Sub-task 7d (conditional — B3 cohort a — Phase 6 share-counter emits):**

Per §"Cross-phase telemetry — cohort split (B3 reshape)", Phase 8
owns the two `composer.share.*` counter emits that earlier drafts
incorrectly attributed to Phase 6. Phase 6 is mid-delivery and
cannot be amended; Phase 8 wires the emits at Phase 6's verify-failure
and expiry-hit code sites.

Probe (re-run from Task 0 for branch decision):

```bash
grep -rn 'verify_token\|verify_share_token' src/elspeth/web/
```

- **Case A — probe hits (Phase 6 token-verify path has shipped):**
  Locate the verify-failure branch (the path returning 401/403 from
  `ShareableReviewService.verify_token` or its successor symbol;
  exact name to be determined at execution time). Emit:

  ```python
  telemetry.share_token_verify_failure_total.add(1)
  ```

  Locate the expiry-hit branch (the path returning a "link expired"
  response). Emit:

  ```python
  telemetry.share_link_expiry_hit_total.add(1)
  ```

  Both emits are operational telemetry. **No new audit event is
  created.** Per CLAUDE.md §"Telemetry and Logging", the superset
  exception for non-decision reads applies: verify-failure and
  expiry-hit are read-path operational failures (`resolve_token` is
  signature-math + payload-store read; failure is not a decision),
  so a telemetry-only signal is the correct channel. Phase 6A's
  plan does not enumerate a verify-failure audit event under its
  §"Audit-event recording" section (R5: the absence in the
  enumerated list is the primary evidence; 19a:117's "No DB
  round-trip on verify" supports the absence indirectly — no DB
  session means no audit row could have been written at that
  branch); Phase 8 does not retroactively claim one. This mirrors the framing used
  for cohort (b2) below (audit-readiness fetch failure, Sub-task
  7f) — both are non-decision read-path failures on shipped or
  in-flight endpoints.

- **Case B — probe misses (Phase 6 lands during or after Phase 8):**
  Documented no-op. The two counters remain in the container (added
  in Step 5) and are filed as a Phase 9 follow-up in
  `21-phase-9-followups.md`. Surface the timing collision to the
  operator at Phase 8 close per §"Probe safety policy".

Trust tier: token-verify failure is a Tier-3 boundary event — an
external presenter offered a token and validation rejected it.
Incrementing an aggregate counter at this boundary is operational
telemetry only, no row-level identity recorded.

**(Q7 — conditional emit tests for Sub-task 7d):** add two
route-level tests, both gated on the Phase 6 token-verify symbol
being importable. Same pattern as Sub-task 7b's Q7 test (probe at
test-module import time; `pytest.mark.skipif` on a module-level
`_PHASE_6_TOKEN_VERIFY_SHIPPED` boolean):

```python
try:
    from elspeth.web.shareable_reviews.service import ShareableReviewService  # noqa: F401
    _PHASE_6_TOKEN_VERIFY_SHIPPED = True
except ImportError:
    _PHASE_6_TOKEN_VERIFY_SHIPPED = False


@pytest.mark.skipif(
    not _PHASE_6_TOKEN_VERIFY_SHIPPED,
    reason="Phase 6 token-verify path not shipped; Sub-task 7d "
           "is a documented no-op per Task 0 probe.",
)
def test_verify_failure_emits_share_token_verify_failure_total(...):
    # Drive a verify call with an invalid token; assert 401/403.
    ...
    assert observed_value(
        sessions_telemetry.share_token_verify_failure_total
    ) == 1


@pytest.mark.skipif(
    not _PHASE_6_TOKEN_VERIFY_SHIPPED,
    reason="Phase 6 token-verify path not shipped; Sub-task 7d "
           "is a documented no-op per Task 0 probe.",
)
def test_expiry_hit_emits_share_link_expiry_hit_total(...):
    # Drive a verify call with an expired token; assert the
    # expired-link response.
    ...
    assert observed_value(
        sessions_telemetry.share_link_expiry_hit_total
    ) == 1
```

Verify the import symbol against Phase 6's actual export at
implementation time (`ShareableReviewService` may be a successor
symbol). The two tests are kept separate (not parametrised)
because they exercise distinct code paths in the verify route
(signature-rejection vs. timestamp-rejection) and one regressing
without the other is a real signal.

**Sub-task 7e (conditional — B3 cohort b1 — Phase 5b interpretation opt-out):**

Per §"Cross-phase telemetry — cohort split (B3 reshape)", Phase 8
owns the `composer.interpretation.opt_out_total` emit. Phase 5b is
shipped (memory: `project_phase5b_shipped`); the opt-out fact is
already in the Landscape as
`interpretation_source='auto_interpreted_opt_out'` rows. The counter
is a pure aggregate over those rows and satisfies the audit-superset
rule.

Probe (re-run from Task 0):

```bash
grep -rn 'auto_interpreted_opt_out' src/elspeth/web/
```

- **Case A — probe hits:** locate the opt-out route — most likely
  in `src/elspeth/web/sessions/routes.py` or
  `src/elspeth/web/composer/...`; exact site to be located at
  execution time by following the audit-source string. Emit at the
  point where the audit row is committed:

  ```python
  telemetry.interpretation_opt_out_total.add(1)
  ```

  No attributes; aggregate-only. Audit primacy is preserved — the
  audit row already exists; the counter is derived from it.

- **Case B — probe misses:** Phase 5b should have shipped per memory.
  If the probe misses, surface to operator (memory may be stale or
  the symbol may have been renamed). Do **not** emit a synthetic
  counter; documented no-op + Phase 9 follow-up.

Trust tier: the emit fires on a Tier-2 boundary (post-source
pipeline data the operator has just confirmed); the audit row is
Tier-1 and is already written by Phase 5b's existing code path.

**(Q7 — conditional emit test for Sub-task 7e):** add a route-level
test gated on the Phase 5b opt-out audit-source string being
present in the codebase. Per memory `project_phase5b_shipped` the
symbol should exist; the skip pathway preserves test-suite green
behaviour if the memory is stale or the symbol has been renamed.
The probe here is a content check rather than a symbol import
because `'auto_interpreted_opt_out'` is a string literal in an
audit row construction, not an exported identifier:

```python
import pathlib

_WEB_ROOT = pathlib.Path("src/elspeth/web")
_PHASE_5B_OPT_OUT_PRESENT = any(
    "auto_interpreted_opt_out" in p.read_text()
    for p in _WEB_ROOT.rglob("*.py")
    if p.is_file()
)


@pytest.mark.skipif(
    not _PHASE_5B_OPT_OUT_PRESENT,
    reason="Phase 5b opt-out audit-source not present; Sub-task 7e "
           "is a documented no-op per Task 0 probe.",
)
def test_interpretation_opt_out_emits_counter(...):
    # Drive the opt-out route so a row with
    # interpretation_source='auto_interpreted_opt_out' is committed.
    ...
    assert observed_value(
        sessions_telemetry.interpretation_opt_out_total
    ) == 1
```

Verify the route entry point at implementation time. If the probe
finds the audit-source string but no route can drive it (e.g. the
string is in a fixture, not a live code path), surface to the
operator — Phase 5b's surface may have been refactored.

**Sub-task 7f (conditional — B3 cohort b2 — Phase 2C audit-readiness fetch failure):**

Per §"Cross-phase telemetry — cohort split (B3 reshape)", Phase 8
owns the `composer.audit.fetch_failure_total` emit. Phase 2C is
shipped (memory: `project_phase2c_implementation_complete.md`). The
underlying event is a non-decision read failure on the audit-readiness
endpoint, so the CLAUDE.md superset exception for non-decisions
applies: no new audit event is required; the counter is a telemetry-only
signal of read-path health.

Probe (re-run from Task 0):

```bash
grep -rn 'audit_readiness\|composer/audit/readiness' src/elspeth/web/
```

- **Case A — probe hits:** locate the audit-readiness endpoint and
  its fetch-failure branch (the path returning a 500 or surfacing an
  exception from the audit-readiness service). Emit at that branch:

  ```python
  telemetry.audit_fetch_failure_total.add(1)
  ```

  No attributes; aggregate-only. **No new audit event** — this is
  explicitly a telemetry-only signal under the non-decision exception
  in CLAUDE.md §"Telemetry and Logging".

- **Case B — probe misses:** documented no-op. Counter remains in
  the container; emit deferred to Phase 9.

Trust tier: the fetch-failure event is an internal operational
failure on a read path, not a decision; telemetry-only is the correct
channel per the superset exception.

**(Q7 — conditional emit test for Sub-task 7f):** add a route-level
test gated on the Phase 2C audit-readiness endpoint being mounted.
Per memory `project_phase2c_implementation_complete.md` the
endpoint should be present; the skip pathway exists for symmetry
with the other Q7 tests and to keep the test file robust against
sibling-phase renames. Recommended probe: a FastAPI route-table
inspection, because the endpoint is wired by route registration
rather than a stand-alone exported function:

```python
try:
    from elspeth.web.app import app as _composer_app  # exact import to verify
    _PHASE_2C_AUDIT_READINESS_MOUNTED = any(
        getattr(r, "path", "").endswith("/audit/readiness")
        for r in _composer_app.routes
    )
except ImportError:
    _PHASE_2C_AUDIT_READINESS_MOUNTED = False


@pytest.mark.skipif(
    not _PHASE_2C_AUDIT_READINESS_MOUNTED,
    reason="Phase 2C audit-readiness endpoint not mounted; "
           "Sub-task 7f is a documented no-op per Task 0 probe.",
)
def test_audit_readiness_fetch_failure_emits_counter(...):
    # Force the audit-readiness service to raise (mock the
    # underlying readiness query to throw RuntimeError), then
    # GET /audit/readiness and assert 500. The exception branch
    # is where the emit lives.
    ...
    assert observed_value(
        sessions_telemetry.audit_fetch_failure_total
    ) == 1
```

Verify the FastAPI app import path and the route path at
implementation time — the project may register routes under a
prefix that changes the trailing-segment match.

- [ ] **Step 8: Run the full test suite and commit**

Run `.venv/bin/python -m pytest src/elspeth/web/composer/
src/elspeth/web/sessions/` and `cd src/elspeth/web/frontend && npm
test -- --run`. All green. Commit: `feat(composer): wire Phase 8
telemetry helpers + harvest deferred markers (Phase 8.1)`.

---

## Task 2: Mode-related aggregate metrics

**Files:**

- Modify: `src/elspeth/web/sessions/routes.py` (Task 1 wired the
  opt-out/opt-in events; this task adds the session-switch event).
- Modify: `src/elspeth/web/composer/telemetry_phase8.py` (no new
  helpers needed; reuses Task 1's `record_session_switched`).
- Modify: `src/elspeth/web/composer/telemetry_phase8_test.py` (new
  test for the route-level emit choreography).

**Why separate:** wires the four aggregate metrics named in design doc 05 (opt-out rate, completion rate, per-mode session-switch rate) using helpers from Task 1; the fourth (completion rate) is wired via Phase 6 or deferred.

### Probe step

- [ ] **Step 1: Probe — does a per-session mode switch endpoint exist?**

```bash
grep -n 'trust_mode\|session.*mode\|switch.*mode\|mode.*switch' \
  /home/john/elspeth/src/elspeth/web/sessions/routes.py | head -20
```

Phase 1A introduced per-session `trust_mode` on the existing
`update_composer_preferences` route. A **mid-flow session switch**
is the same endpoint with a different request shape. If the PATCH
already handles session-level trust-mode changes (it does, per
Phase 1A), Task 2 only needs to differentiate "first-time set"
from "user-initiated change" telemetry-wise.

If Phase 1A does not differentiate, the conservative call is to
emit `session_switched_total` **whenever the PATCH changes the
mode**, and accept that the very first transition out of the
backend default counts as a switch. This matches the design-doc-10
intent ("per-mode session-switch rate") because the per-mode rate
is a ratio whose denominator is total sessions.

### TDD steps

- [ ] **Step 2: Write the failing test — extend
  telemetry_phase8_test.py**

Two route-level tests, using whatever fixture pattern Phase 1A's
route test file established (mirror its `sessions_app_client` and
`sessions_telemetry` fixtures):

- `test_route_emits_session_switched_on_mode_change` — create a
  session with `trust_mode="guided"`, PATCH to `"freeform"`, assert
  200, assert `observed_value(sessions_telemetry.session_switched_total)
  == 1`, assert recorded attributes are exactly `{"from_mode":
  "guided", "to_mode": "freeform"}`.
- `test_route_does_not_emit_session_switched_when_mode_unchanged` —
  same setup, PATCH a non-mode field (e.g. `density_default`),
  assert 200, assert counter remains at 0.
- **(Q4 — combined-PATCH coverage)**
  `test_route_emits_session_switched_once_when_mode_and_density_both_change` —
  create a session whose prior `trust_mode == "explicit_approve"`,
  then PATCH the body `{"trust_mode": "auto_commit",
  "density_default": "compact"}` in a single request. Assert 200;
  assert
  `observed_value(sessions_telemetry.session_switched_total) == 1`
  (NOT 0, NOT 2); assert recorded attributes are exactly
  `{"from_mode": "explicit_approve", "to_mode": "auto_commit"}`.
  Rationale: the emit is gated on `prior.trust_mode !=
  current.trust_mode`, so co-changed fields must neither suppress
  the emit (counter at 0 would mean the route handler is checking
  the wrong predicate, e.g. "only mode changed AND nothing else
  changed") nor double-count it (counter at 2 would mean the emit
  is duplicated per PATCHed field rather than per PATCH request).
  The exact-attribute assertion also guards the per-session trust
  vocabulary contract from §"Vocabulary discipline (B1-r2 —
  load-bearing)".
- **(Q8 — B1 audit-payload contract test)**
  `test_trust_mode_changed_audit_event_records_prior_and_new_state` —
  set up a session with `trust_mode == "explicit_approve"`. PATCH
  `/api/sessions/{session_id}/composer/preferences` with body
  `{"trust_mode": "auto_commit"}`. Assert 200. Then query
  `proposal_events_table` for rows with
  `event_type == "trust_mode.changed"` for that session, ordered by
  insertion order; load the most recent row's JSON `payload`.
  Assert `payload["prior_trust_mode"] == "explicit_approve"` AND
  `payload["trust_mode"] == "auto_commit"` — **both** keys present
  with **both** values. Rationale: this is the test that asserts
  the B1 audit-payload extension (per §"Audit-payload precondition
  (B1 — load-bearing)") actually landed. Without it, the B1
  precondition is prose-only and a future PR could silently revert
  the audit-payload extension while leaving `record_session_switched`
  emitting transition attributes — re-inverting audit primacy. This
  test lives alongside the `session_switched_total` route tests
  because they share fixtures (same PATCH route, same session
  setup) and because the per-session emit is the consumer of the
  B1 extension; Sub-task 7a (account-level) deliberately does not
  produce a `trust_mode.changed` event per §B2.b, so the test
  belongs here, not under Sub-task 7a.

Run: expected fail (no emit at route level yet).

- [ ] **Step 3: Implement the route emit**

In `update_composer_preferences` in
`src/elspeth/web/sessions/routes.py` (the **per-session**
`update_composer_preferences` route handler, mounted at
`PATCH /api/sessions/{session_id}/composer/preferences` — **NOT**
the account-level `update_preferences` handler in
`web/preferences/routes.py` that Task 1 Sub-task 7a wires), after
the audit event persists and **before** the response is constructed:

```python
if prior.trust_mode != current.trust_mode:
    # from_mode / to_mode are trust_mode values per
    # src/elspeth/web/sessions/models.py:150 (CHECK constraint:
    # IN ('explicit_approve', 'auto_commit')). These match the
    # per-session helper's _SessionTrustMode Literal exactly;
    # see §"Vocabulary discipline (B1-r2 — load-bearing)" for
    # why this differs from the account-level helpers'
    # ('guided' / 'freeform') vocabulary.
    record_session_switched(
        telemetry,
        from_mode=prior.trust_mode,
        to_mode=current.trust_mode,
    )
```

**Precondition (B1):** the per-session `trust_mode.changed` audit
event must already record both `prior_trust_mode` and `trust_mode`
in its JSON payload (covered by §"Audit-payload precondition (B1 —
load-bearing)"; that subsection's prose names this event
explicitly). Without it, this emit would make telemetry a superset
of audit-recorded reality.

**Precondition (B2):** see §"Service signature precondition (B2 —
load-bearing)" — the per-session
`sessions_service.update_composer_preferences` must return both the
prior and the current record so the route handler has
`prior.trust_mode` in scope. Without this reshape, the branch above
cannot compile.

This emit runs **alongside** the account-level opt-out/opt-in emit
from Task 1 Sub-task 7a; the two live in different route files and
fire on different PATCH endpoints. They are complementary:
account-level opt-out/opt-in is the user-level "defaulting away
from guided" signal (one row in the user-preferences table);
session-switched is the session-level "this session changed mode"
signal (one row in the session-events table). Both increment when
both transitions happen; neither subsumes the other.

- [ ] **Step 4: Run tests + commit**

Run `.venv/bin/python -m pytest src/elspeth/web/sessions/routes_test.py
src/elspeth/web/composer/telemetry_phase8_test.py -v`. Commit:
`feat(composer): emit session_switched_total on mode change (Phase 8.2)`.

---

## Task 3: Templates → README's "Example Use Cases" mapping

**Files:**

- Create: `src/elspeth/web/frontend/src/components/chat/templates_data.ts`
- Create: `src/elspeth/web/frontend/src/components/chat/templates_data.test.ts`
- Modify: `src/elspeth/web/frontend/src/components/chat/TemplateCards.tsx`
- Modify: `src/elspeth/web/frontend/src/components/chat/TemplateCards.test.tsx`

**Why separate:** keeps the README → templates mapping testable in isolation; future README edits touch one file.

### Read step

- [ ] **Step 1: Read the README "Example Use Cases" table**

Read `/home/john/elspeth/README.md` lines 560-571. The table has
six rows; each row maps to one template card:

| Domain | Sense | Decide | Act |
|---|---|---|---|
| Tender Evaluation | CSV of submissions | LLM classification + safety gates | Results CSV, abuse review queue |
| Document QA | PDF/text blobs | LLM extraction, rubric checks, statistical summaries | Annotated outputs, exception queue |
| Weather Monitoring | Sensor API feed | Threshold + ML anomaly detection | Routine log, warning, emergency alert |
| Satellite Operations | Telemetry stream | Anomaly classifier | Routine log, investigation ticket |
| Financial Compliance | Transaction feed | Rules engine + ML fraud detection | Approved, flagged, blocked |
| Content Moderation | User submissions | Safety classifier | Published, human review, rejected |

Each becomes a card. The card shape adds a **seed prompt** field
that the composer chat will use when the user clicks the card.
Seed prompts are written so the dynamic-source-from-chat affordance
(Phase 5a) can consume them — they describe a small concrete
scenario the composer can build a 1-row dynamic source from.

### TDD steps

- [ ] **Step 2: Write the failing test —
  `templates_data.test.ts`**

Create `/home/john/elspeth/src/elspeth/web/frontend/src/components/chat/templates_data.test.ts`:

```ts
import { describe, expect, it } from "vitest";
import { TEMPLATES, type ExampleUseCase } from "./templates_data";

describe("templates_data — README Example Use Cases mapping", () => {
  it("contains exactly six audit-domain exemplars", () => {
    expect(TEMPLATES).toHaveLength(6);
  });

  it("every template has the four READMEs-table columns", () => {
    for (const t of TEMPLATES) {
      expect(t.domain).toBeTruthy();
      expect(t.sense).toBeTruthy();
      expect(t.decide).toBeTruthy();
      expect(t.act).toBeTruthy();
    }
  });

  it("every template has a seed_prompt suitable for chat dispatch", () => {
    for (const t of TEMPLATES) {
      expect(t.seed_prompt.length).toBeGreaterThan(40);
      expect(t.seed_prompt.length).toBeLessThan(400);
    }
  });

  it("every template has a recommended_starting_point", () => {
    for (const t of TEMPLATES) {
      expect(["dynamic_source_from_chat", "csv_upload", "api_source"]).toContain(
        t.recommended_starting_point,
      );
    }
  });

  it("ids are stable and unique", () => {
    const ids = TEMPLATES.map((t) => t.id);
    expect(new Set(ids).size).toBe(ids.length);
    // Stability test: hard-coded snapshot so a future PR that
    // rearranges the array breaks here, not in a downstream test.
    expect(ids).toEqual([
      "tender-evaluation",
      "document-qa",
      "weather-monitoring",
      "satellite-operations",
      "financial-compliance",
      "content-moderation",
    ]);
  });

  it("the type is exported (used by TemplateCards.tsx)", () => {
    const sample: ExampleUseCase = TEMPLATES[0];
    expect(sample.id).toBeTruthy();
  });
});
```

Run: expected fail (module doesn't exist).

- [ ] **Step 3: Implement `templates_data.ts`**

Create `/home/john/elspeth/src/elspeth/web/frontend/src/components/chat/templates_data.ts`.
Module-level docstring should call out: "Audit-domain exemplars sourced
from README.md lines 560-571; the empty-state chat consumes this.
Update discipline — if README.md's table changes, update this file and
the snapshot in `templates_data.test.ts` in the same PR."

Types:

- `RecommendedStartingPoint = "dynamic_source_from_chat" | "csv_upload"
  | "api_source"`
- `ExampleUseCase` with fields: `id`, `domain`, `description`,
  `sense`, `decide`, `act`, `seed_prompt`, `recommended_starting_point`,
  `icon` (all strings except the recommended-starting-point literal).

`TEMPLATES: ReadonlyArray<ExampleUseCase>` — six entries, in this id
order (matches the snapshot test from Step 2):

| id | Domain | Sense | Decide | Act | Recommended start | Icon (codepoint) |
|---|---|---|---|---|---|---|
| `tender-evaluation` | Tender Evaluation | CSV of submissions | LLM classification + safety gates | Results CSV, abuse review queue | `dynamic_source_from_chat` | `\u{1F4DD}` |
| `document-qa` | Document QA | PDF/text blobs | LLM extraction, rubric checks, statistical summaries | Annotated outputs, exception queue | `csv_upload` | `\u{1F4C4}` |
| `weather-monitoring` | Weather Monitoring | Sensor API feed | Threshold + ML anomaly detection | Routine log, warning, emergency alert | `dynamic_source_from_chat` | `\u{1F324}` |
| `satellite-operations` | Satellite Operations | Telemetry stream | Anomaly classifier | Routine log, investigation ticket | `dynamic_source_from_chat` | `\u{1F6F0}` |
| `financial-compliance` | Financial Compliance | Transaction feed | Rules engine + ML fraud detection | Approved, flagged, blocked | `dynamic_source_from_chat` | `\u{1F4B3}` |
| `content-moderation` | Content Moderation | User submissions | Safety classifier | Published, human review, rejected | `dynamic_source_from_chat` | `\u{1F6E1}` |

Seed prompts are short scenario descriptions (40-400 chars per the
test) that the composer can dispatch through dynamic-source-from-chat
to build a 1-row inline source. Example for `tender-evaluation`: "I
want to evaluate three tender submissions. Each has a vendor name, a
price, and a 200-word capability statement. Use an LLM to score each
on capability fit, and route anything mentioning offensive language
to a review queue." Write one such prompt per row, scaled to the
domain's tiniest demonstrable shape (3-10 rows of synthetic data).

Description strings are one-line restatements of the README row
suitable for the card's `<span class="template-card-description">`
slot (e.g. "Score procurement submissions; flag responses that need
human review.").

- [ ] **Step 4: Refactor `TemplateCards.tsx` to consume
  `templates_data.ts`**

Replace the inline `TEMPLATES` array (lines 16-65 of the current
file) with `import { TEMPLATES, type ExampleUseCase } from
"./templates_data"`.

Component-shape changes:

- `TemplateCardsProps.onSelectTemplate` signature gains a second
  argument: `(seedPrompt: string, recommendedStartingPoint:
  ExampleUseCase["recommended_starting_point"]) => void`.
- The heading subtitle changes from "Choose a template…" to a
  one-line statement that names auditability ("ELSPETH builds
  **auditable** pipelines. Start from a domain example below, or
  describe your own pipeline in the chat.").
- The grid's `aria-label` changes from "Pipeline templates" to
  "Example use cases".
- Each `<button>` gains an `aria-label` of
  `` `${template.domain}: ${template.description}` `` so
  screen-reader users get the SDA-irrelevant content first.
- Each card body adds an SDA breakdown after the description: a
  `<dl class="template-card-sda">` with three `<div><dt>…</dt><dd>…</dd></div>`
  rows for Sense, Decide, Act (the README columns).

Update every call-site in the same commit. Find them:

```bash
grep -rn 'onSelectTemplate\|TemplateCards' \
  /home/john/elspeth/src/elspeth/web/frontend/src --include='*.tsx' --include='*.ts'
```

The downstream handler routes the chat dispatch through
dynamic-source-from-chat (Phase 5a) when the recommended starting
point is `"dynamic_source_from_chat"`. If Phase 5a hasn't shipped,
the handler ignores the second argument and that gap is documented
in a Phase 9 follow-up.

- [ ] **Step 5: Update `TemplateCards.test.tsx`**

Adjust the existing tests to expect six cards instead of six
generic ones, and to read the new strings from `templates_data.ts`.
The shape of the test (click → onSelectTemplate called) does not
change.

- [ ] **Step 6: Update CSS — `.template-card-sda` styling**

The new SDA `<dl>` block needs basic styling. Find the existing
`.template-card-description` rule in the CSS module and add:

```css
.template-card-sda {
  margin-top: 0.5rem;
  display: grid;
  grid-template-columns: auto 1fr;
  gap: 0.25rem 0.75rem;
  font-size: 0.75rem;
  color: var(--color-text-muted);
}
.template-card-sda dt {
  font-weight: 600;
}
.template-card-sda dd {
  margin: 0;
}
```

(Adjust to the project's existing CSS variable / module
convention. Confirm path: `grep -l "template-card" src/elspeth/web/frontend/src --include='*.css'`.)

- [ ] **Step 7: Run tests + commit**

`cd src/elspeth/web/frontend && npm test -- --run` — green.
Commit: `feat(composer): replace generic templates with README Example Use Cases (Phase 8.3)`.

---

## Task 4: Session sidebar → header switcher migration

**Files:**

- Read: `src/elspeth/web/frontend/src/components/sessions/HeaderSessionSwitcher.tsx` (existence probe).
- Modify (if it exists): `HeaderSessionSwitcher.tsx` — add filter + archive controls.
- Modify (if migration is ready): `src/elspeth/web/frontend/src/App.tsx` — drop the `SessionSidebar` mount.
- Delete (if migration is ready):
  - `src/elspeth/web/frontend/src/components/sessions/SessionSidebar.tsx`
  - `src/elspeth/web/frontend/src/components/sessions/SessionSidebar.test.tsx`

### Probe step

- [ ] **Step 1: Probe — does HeaderSessionSwitcher exist?**

```bash
ls /home/john/elspeth/src/elspeth/web/frontend/src/components/sessions/HeaderSessionSwitcher.tsx 2>/dev/null
```

**Case A — file exists (Phase 3B shipped):** continue with Steps 2-7
as the *polish + sidebar deletion* path.

**Case B — file does not exist:** Phase 3B hasn't shipped the
switcher yet. Two sub-cases:

- **B-1: Phase 3B is planned but not implemented:** this task is a
  documented no-op. Add a follow-up note to
  `docs/composer/ux-redesign-2026-05/21-phase-9-followups.md`
  (created lazily) saying "Phase 8 Task 4 deferred until Phase 3B
  ships HeaderSessionSwitcher." Stop.
- **B-2: Phase 3B was descoped:** this task implements the
  switcher itself. The scope balloons; surface to the operator
  before continuing. Phase 8 should not silently absorb the
  Phase 3B scope.

If you reach B-2, **stop and surface to the operator**.
[CLAUDE.md](/home/john/elspeth/CLAUDE.md) §"Human Operator
Communication" applies: do not assume silence means consent to
absorb the scope.

### TDD steps (Case A)

- [ ] **Step 2: Read the existing HeaderSessionSwitcher**

```bash
cat /home/john/elspeth/src/elspeth/web/frontend/src/components/sessions/HeaderSessionSwitcher.tsx
```

Inventory what's there:

- A list of sessions (presumably from `sessionStore`).
- A click handler that activates a session.
- A "new session" button somewhere (or it lives in the menu).

What's likely missing (the Phase 8 polish targets):

- A filter input ("Find a session…").
- An archive control (per-session "Archive" verb).
- An archived-sessions toggle ("Show archived").

- [ ] **Step 3: Write the failing test —
  HeaderSessionSwitcher.test.tsx**

Append four tests to the existing file (mirror the project's
`renderWithSessions` fixture pattern). Each test opens the
switcher (`fireEvent.click(screen.getByRole("button", { name:
/sessions/i }))`) before asserting:

- `renders a session filter input` — expects a `textbox` whose
  accessible name matches `/find a session/i`.
- `filters sessions by title (case-insensitive substring)` —
  seeds three sessions (Tender review / Weather monitor / Document
  QA pipeline), types "weather" into the filter, asserts only
  "Weather monitor" remains visible.
- `shows an archive button on each session row` — seeds one
  session, asserts a button named `archive tender review` is
  rendered.
- `hides archived sessions by default; shows them when toggled` —
  seeds an active and an archived session, asserts only the active
  is visible, then clicks the `Show archived` checkbox and asserts
  the archived row appears.
- **(Q9 — backend-archive runtime failure)**
  `surfaces a backend archive failure via the error region and
  preserves the row in the active list` — seed one session named
  "Tender review". Mock `sessionStore.archiveSession` with
  `vi.fn().mockRejectedValue(new Error("backend unavailable"))`
  (or whatever the project's promise-rejection-shape convention
  is). Click the per-row archive button. Await both assertions:
  (a) an element with `role="alert"` (or the project's existing
  error-toast / error-region role — verify against any prior a11y
  pattern in this file; do not invent a new convention) renders
  containing the error text or a recognisable surrogate
  (`/could not archive|backend unavailable|unable to archive/i`);
  (b) the "Tender review" row remains present in the active-sessions
  list (no optimistic-UI divergence: a failed archive must not
  remove the row, because the backend state is unchanged). Without
  this test, the archive action is asserted only on the happy path,
  and an optimistic-UI bug that removes the row on the network call
  rather than on the response would ship undetected.

Run: expected fail.

- [ ] **Step 4: Implement filter + archive controls**

Extend `HeaderSessionSwitcher.tsx`:

- Add a `useState` for the filter text.
- Add a `useState` for `showArchived` (default false).
- Add a `<input type="text" aria-label="Find a session…" />` near
  the top of the dropdown.
- For each session row, add a small `<button aria-label="Archive
  {title}">` that calls `sessionStore.archiveSession(id)`.
- Filter the rendered list: `(s) => (showArchived || !s.archived) &&
  s.title.toLowerCase().includes(filter.toLowerCase())`.
- Add a checkbox: `<input type="checkbox" aria-label="Show
  archived" />`.

`sessionStore.archiveSession` may not exist. If not:

- Probe: `grep -n 'archiveSession\|archived' src/elspeth/web/frontend/src/stores/sessionStore.ts` (NOT `src/state/` — that directory does not exist; the project layout uses `src/stores/`). Note: `archiveSession` already exists at `stores/sessionStore.ts:184` and `api/client.ts:341`, so the "may not exist" branch below is moot in current head.
- If missing, add the action; the backend should already accept
  `PATCH /api/sessions/{id}` with `{archived: true}` (or whatever the
  existing convention is; do not invent a new endpoint).

If the backend doesn't support archival, surface to operator —
don't silently add a new endpoint.

- [ ] **Step 5: Delete the old SessionSidebar**

After the test in Step 3 passes and the new switcher fully
subsumes the sidebar's job:

```bash
rm /home/john/elspeth/src/elspeth/web/frontend/src/components/sessions/SessionSidebar.tsx
rm /home/john/elspeth/src/elspeth/web/frontend/src/components/sessions/SessionSidebar.test.tsx
```

Remove the mount in `App.tsx`. Search for any remaining import or
reference:

```bash
grep -rn 'SessionSidebar' /home/john/elspeth/src/elspeth/web/frontend/src/ 2>/dev/null
```

Expected: zero matches after the deletion. If any non-test file
still imports it, that's a sign the migration isn't complete —
fix before continuing.

- [ ] **Step 6: Update CSS — remove `.session-sidebar` rules**

Search:

```bash
grep -rn 'session-sidebar' /home/john/elspeth/src/elspeth/web/frontend/src/ --include='*.css'
```

Delete every rule. If the rules share a stylesheet with other
sidebar components (`.sidebar`, `.sidebar-resize-handle`), check
each one — only delete the SessionSidebar-specific rules.

- [ ] **Step 7: Run tests + commit**

`cd src/elspeth/web/frontend && npm test -- --run`. Commit:
`feat(composer): retire SessionSidebar; polish HeaderSessionSwitcher (Phase 8.4)`.

If the probe in Step 1 returned **Case B-1**, this task ends as a
documented no-op — record it in `21-phase-9-followups.md` and skip
the commit.

---

## Task 5: Keyboard-shortcut audit and reorganisation

**Files:**

- Read + modify: `src/elspeth/web/frontend/src/App.tsx` (the
  `handleKeyDown` keyboard handler block inside the `useEffect`
  that wires `document.addEventListener("keydown", …)`; verify
  location at implementation time by grepping for `handleKeyDown`).
- Modify: `src/elspeth/web/frontend/src/components/common/ShortcutsHelp.tsx`
- Modify: `src/elspeth/web/frontend/src/components/common/ShortcutsHelp.test.tsx`

### Inventory step

- [ ] **Step 1: Inventory existing shortcuts**

The current `handleKeyDown` function in `App.tsx` defines:

| Combo | Action | Category |
|---|---|---|
| `Ctrl/Cmd+K` | Command palette | Navigation |
| `Ctrl/Cmd+Shift+P` | Open plugin catalog | Reference |
| `Ctrl/Cmd+N` | New session | Actions |
| `Ctrl/Cmd+/` | Focus chat input | Navigation |
| `Alt+1` | Switch inspector tab: Spec | Navigation (obsolete post-Phase-3) |
| `Alt+2` | Switch inspector tab: Graph | Navigation (obsolete post-Phase-3) |
| `Alt+3` | Switch inspector tab: YAML | Navigation (obsolete post-Phase-3) |
| `Alt+4` | Switch inspector tab: Runs | Navigation (obsolete post-Phase-3) |
| `Ctrl/Cmd+Shift+V` | Validate pipeline | Actions (obsolete post-Phase-2: subsumed into audit-readiness panel) |
| `Ctrl/Cmd+E` | Execute pipeline | Actions |
| `?` | Show keyboard shortcuts | Reference |
| `Escape` | Close dialog or drawer | Editing |

### Probe step

- [ ] **Step 2: Probe — are the Alt+1-4 shortcuts still valid?**

Phase 3 removed the Spec / Runs / Graph / YAML tabs from the
inspector. Check what `SWITCH_TAB_EVENT` does post-Phase-3:

```bash
grep -rn 'SWITCH_TAB_EVENT\|SWITCH_TAB' /home/john/elspeth/src/elspeth/web/frontend/src/ | head -10
```

If the event is no longer subscribed-to by any inspector
component (Phase 3 dropped the listener), the Alt+1-4 shortcuts
are dead code — delete them and rebind those slots if needed.

Similarly, probe whether `Ctrl+Shift+V` (validate) still has a
consumer post-Phase 2:

```bash
grep -rn 'useExecutionStore.*validate\|validate(' /home/john/elspeth/src/elspeth/web/frontend/src/ | head -10
```

If Phase 2 subsumed the standalone Validate button into the
audit-readiness panel, the shortcut might still drive the
panel's "validate now" action. Keep it if the consumer exists;
delete if it doesn't.

### TDD steps

- [ ] **Step 3: Write the failing test —
  ShortcutsHelp.test.tsx (reorganised)**

Replace the current flat-list assertions with grouped assertions.
Six test cases:

- `renders an Actions group` / `…a Reference group` / `…a Navigation
  group` / `…an Editing group` — each asserts a `heading` element
  whose accessible name matches the group's name.
- `each shortcut is associated with exactly one group` — render,
  collect every `<kbd>` whose text starts with `Ctrl`, `?`,
  `Escape`, or `Alt`, walk each up to its `closest("section")`,
  assert the section's `aria-label` is one of the four group
  names.
- `the obsolete Alt+1-4 inspector-tab shortcuts are gone` —
  asserts no text matches `/Alt\+1-4/` or `/Spec\/Graph\/YAML\/Runs/`.

Run: expected fail.

- [ ] **Step 4: Implement the grouped ShortcutsHelp**

Replace the flat `SHORTCUTS` constant with a `GROUPS` array of
`{name, items: {keys, action}[]}` objects. Distribution:

- **Actions**: `Ctrl+N` New session, `Ctrl+E` Run pipeline.
- **Navigation**: `Ctrl+K` Command palette, `Ctrl+/` Focus chat input.
- **Reference**: `Ctrl+Shift+P` Open plugin catalog, `?` Keyboard
  shortcuts.
- **Editing**: `Escape` Close dialog or drawer.

If Step 2 found the validate shortcut still has a consumer, add
`Ctrl+Shift+V — Validate pipeline` to **Actions**; otherwise delete
the binding from both `App.tsx` and `ShortcutsHelp`.

Render each group as a `<section aria-label={group.name}
className="shortcuts-group">` containing an `<h3>` heading and a
`<dl class="shortcuts-list">` of the existing `<dt><kbd>…</kbd></dt><dd>…</dd>`
shape. The DOM shape preserves the existing `useFocusTrap` and
modal backdrop unchanged.

- [ ] **Step 5: Drop obsolete shortcuts from App.tsx**

If Step 2 found Alt+1-4 dead post-Phase-3, delete the
`Alt+1`/`Alt+2`/`Alt+3`/`Alt+4` branch (the one dispatching
`SWITCH_TAB_EVENT`) from `handleKeyDown` in `App.tsx`. Same for
`Ctrl+Shift+V` if its consumer is gone — locate by grepping
`handleKeyDown` for the `e.key === "v"` branch.

- [ ] **Step 6: Verify no consumer regression**

```bash
cd src/elspeth/web/frontend && npm test -- --run
```

Pay close attention to any test that mocked `SWITCH_TAB_EVENT` —
those tests may be testing Phase-3-removed surface and should be
deleted, not patched.

- [ ] **Step 7: Commit**

`polish(composer): regroup keyboard shortcuts by category (Phase 8.5)`.

---

## Task 6: Tutorial-replay affordance in settings

**Files:**

- Modify: `src/elspeth/web/frontend/src/components/settings/ComposerPreferencesPanel.tsx`
- Create: `src/elspeth/web/frontend/src/components/settings/TutorialReplayButton.tsx`
- Create: `src/elspeth/web/frontend/src/components/settings/TutorialReplayButton.test.tsx`
- Modify: `src/elspeth/web/frontend/src/api/client.ts` (extend `updateComposerPreferences`'s request type to allow `tutorial_completed_at: string | null`; no new function added — see Step 5).
- Modify: `src/elspeth/web/frontend/src/stores/preferencesStore.ts` (expose `replayTutorial()`).
- Modify: `tests/integration/web/test_preferences_routes.py` (two retake-specific boundary tests — see Step 3 Q5 block: `test_retake_patch_with_explicit_null_clears_tutorial`, `test_retake_patch_distinguishes_absent_from_null`).
- Modify: `src/elspeth/web/composer/telemetry_phase8.py` — Phase 9 follow-up; Decision 2 Option C defers counter emission, so this entry remains only as the structural placeholder for the deferred replay-counter helper (see Step 7).

### Probe step

- [ ] **Step 1: Probe — does the tutorial column exist?**

```bash
grep -rn 'tutorial_completed_at\|tutorialCompletedAt\|tutorialCompleted' \
  /home/john/elspeth/src/elspeth/web/ --include='*.py' --include='*.ts' --include='*.tsx'
```

**Case A — column exists (Phase 4 shipped):** continue with Steps 2-8.

**Case B — column does not exist:** Phase 4 has not shipped the
hello-world tutorial yet. The replay button has nothing to clear.

- **B-1:** Add a placeholder UI that says "Tutorial replay
  available once the tutorial is added in a future release" and
  do not implement the click handler. Surface to operator —
  this might or might not be the right call.
- **B-2 (recommended):** Skip Task 6 entirely and record it as
  a Phase 9 follow-up. Do not ship a no-op button.

If you reach Case B, **stop and surface to operator**.

### TDD steps (Case A)

- [ ] **Step 2: Confirm the wire shape**

Phase 4 ships `tutorial_completed_at: datetime | None` on the
`user_preferences` row and exposes it on GET/PATCH
`/api/composer-preferences`. See
[21a1-phase-4-backend-part-1.md](21a1-phase-4-backend-part-1.md) §"Cross-plan
contract — `tutorial_completed_at` PATCH semantics" for the
canonical statement of the contract Task 6 relies on. Confirm:

```bash
grep -n 'tutorial_completed_at' \
  /home/john/elspeth/src/elspeth/web/preferences/models.py \
  /home/john/elspeth/src/elspeth/web/preferences/service.py \
  /home/john/elspeth/src/elspeth/web/sessions/models.py
```

The replay action is `PATCH /api/composer-preferences` with body
`{"tutorial_completed_at": null}` — explicit `null` clears the
column per Phase 4's three-state PATCH semantics (absent =
preserve; datetime = write; null = clear). No new endpoint needed.

- [ ] **Step 3: Write the failing test —
  TutorialReplayButton.test.tsx**

Four test cases (mirror the project's `renderWithPreferencesStore`
fixture pattern):

- `renders a button labelled 'Replay hello-world tutorial'` —
  `getByRole("button", { name: /Replay hello-world tutorial/i })`.
- `clicking dispatches the replay action on the preferences store` —
  inject a `vi.fn()` for `replayTutorial`, click, assert called
  exactly once.
- `disables the button while the action is in flight` — inject a
  never-resolving promise, click, await `toBeDisabled`.
- `re-enables and shows confirmation when complete` — inject a
  resolved promise, click, await visibility of the
  `Tutorial will replay on next sign-in` confirmation string.

**(Q5 — Pydantic boundary tests for `tutorial_completed_at`):** the
Tier-3 boundary tests for the field live in Phase 4 (see
[21a1-phase-4-backend-part-1.md](21a1-phase-4-backend-part-1.md) Task 2 and Task 4):
`tutorial_completed_at` is `datetime | None`; the model rejects
non-datetime, non-null values at the boundary with 422. Phase 8
Task 6 adds **two** retake-specific boundary tests on top of
Phase 4's coverage, focused on the explicit-null retake path:

- `test_retake_patch_with_explicit_null_clears_tutorial` — PATCH
  `/api/composer-preferences` with body
  `{"tutorial_completed_at": null}` after a prior PATCH set the
  field. Assert `response.status_code == 200`; assert
  `response.json()["tutorial_completed_at"] is None`. This is the
  Phase 8 retake happy-path.
- `test_retake_patch_distinguishes_absent_from_null` — two
  back-to-back PATCHes from a starting state where the field is
  set: (a) PATCH with body `{"default_mode": "freeform"}` (field
  absent from payload) → assert `tutorial_completed_at` is
  preserved; (b) PATCH with body `{"tutorial_completed_at": null}`
  (field present and null) → assert the column is cleared. This
  pins the three-state contract co-owned with Phase 4.

Phase 4 already covers boundary rejection for non-datetime,
non-null values (`"yesterday"` etc.). No need to duplicate that
coverage here; Phase 8 only owns the retake-specific paths.

Run: expected fail (Task 6 is not yet implemented).

- [ ] **Step 4: Implement `TutorialReplayButton.tsx`**

Component shape (functional component):

- Pull `replayTutorial` from `usePreferencesStore`.
- Two `useState` slots: `busy: boolean`, `done: boolean`.
- `onClick`: set busy=true, clear done, `await replayTutorial()`,
  set done=true, `finally` set busy=false.
- Render a `<div className="tutorial-replay-row">` containing:
  - A `<button type="button" disabled={busy} aria-describedby=
    "tutorial-replay-help">` labelled "Replay hello-world tutorial".
  - A `<p id="tutorial-replay-help" className="form-help-text">`
    with the explanatory copy ("Reshows the introductory tutorial
    on your next sign-in. Your existing sessions are not
    affected.").
  - Conditionally (`{done && ...}`) a `<p role="status"
    className="form-success-text">` with the confirmation copy.

- [ ] **Step 5: Add `replayTutorial` to `preferencesStore`**

In the Zustand store, add a `replayTutorial` action that calls
`api.updateComposerPreferences({ tutorial_completed_at: null })`
then re-invokes `get().bootstrap()` so the cached row reflects the
freshly-cleared column without a race. Phase 4's PATCH contract
treats explicit `null` as "write NULL to the column" (the retake
state); see §"Cross-plan contract" in
[21a1-phase-4-backend-part-1.md](21a1-phase-4-backend-part-1.md).

In `api/client.ts` (where `updateComposerPreferences` actually lives — no separate `api/preferences.ts` module exists), verify
`updateComposerPreferences`'s body type still includes
`tutorial_completed_at: string | null` as Phase 4 ships it (see
[21a1-phase-4-backend-part-1.md](21a1-phase-4-backend-part-1.md) §"Cross-plan
contract — `tutorial_completed_at` PATCH semantics"). If the type
union has regressed, fix Phase 4 — do not patch around it in Phase
8. Ownership stays with Phase 4; Phase 8 verifies.

Do **not** add a separate `clearTutorialCompleted` store action.
The same `updateComposerPreferences` PATCH client function is the
correct surface for both finalisation (writes a timestamp) and
retake (writes null); a parallel action would split a single
shared contract into two and is forbidden.

- [ ] **Step 6: Mount the button in `ComposerPreferencesPanel`**

Add near the bottom of the panel's JSX a `<section
aria-labelledby="tutorial-section-heading">` with an `<h3
id="tutorial-section-heading">Tutorial</h3>` and the
`<TutorialReplayButton />` mount.

- [ ] ~~**Step 7: Emit the replay counter**~~

**DEFERRED to Phase 9** per the Decision 2 resolution above (Option
C). Phase 8 does not emit `composer.tutorial.replayed_total` and
does not add the Q6 transition-edge backend tests for it. The
boundary question (audit-row vs telemetry-only for a user-write-
intent event) is recorded in `21-phase-9-followups.md` so Phase 9's
design pass cannot drop it on the floor.

What this means concretely for Task 6:

- The PATCH-handler `prior → current` transition detection logic
  does **not** ship in 8c. Task 6 stops at Step 6 (button mounted +
  PATCH call wired); the backend handler accepts the PATCH (body
  `{"tutorial_completed_at": null}`) and writes NULL to the column,
  exactly as Phase 4's three-state PATCH contract already permits.
- The §B2 service-signature reshape is **still required** for
  Task 6's UI to function (the button reads `tutorial_completed_at`
  via `bootstrap()` and PATCHes `{"tutorial_completed_at": null}`
  via the existing endpoint), but the *route-handler* consumer of
  the `prior` record is reduced to the two opt-out / opt-in cases
  — the tutorial-replay dependent on §"Service signature
  precondition (B2 — load-bearing)" is removed.
- The three Q6 tests for `tutorial_replayed_total` are deleted.
  Task 6 retains only its UI-render, store-action, and PATCH-
  call tests (Steps 3-6) plus the two retake-specific Pydantic
  boundary tests (Step 3, see Q5 block above).

**Audit-emit boundary (Phase 9 design input).** The retake PATCH
clears `tutorial_completed_at` and thereby destroys the prior
completion timestamp. That destruction is itself audit-worthy:
"user X had completed the tutorial at time T1 and then retook at
time T2" is the kind of fact an auditor would expect the Landscape
to recover. Phase 8 does not yet emit this audit event (per the
Decision 2 deferral), but Phase 9's design pass should consider:
(a) where the audit event lives — `preferences/service.py` versus
the route handler in `preferences/routes.py`; (b) whether the
prior timestamp is captured inside the same transaction as the
clear (mandatory, to avoid TOCTOU); (c) whether the audit event
is a new event type or piggybacks on the existing
`composer.preferences.patch_total` emission point. The §B2
service-signature reshape is a precondition (it makes `prior` and
`current` available at the emit site); the Phase 9 work is to
decide between Landscape-row and telemetry-counter and, if
Landscape-row, the schema. Recorded as a Phase 9 follow-up in
`21-phase-9-followups.md`.

- [ ] **Step 7 (was Step 8): Run tests + commit**

`cd src/elspeth/web/frontend && npm test -- --run`. The backend
pytest run for `src/elspeth/web/preferences/` is no longer load-
bearing for Task 6 (the backend handler did not change beyond
what Phase 4 already shipped), but run it anyway as a safety
check. Commit: `feat(composer): add tutorial-replay affordance
in settings (Phase 8.6, UI-only — counter deferred to Phase 9)`.

---

## Task 7: Accessibility audit (axe-core)

**Files:**

- Modify: `src/elspeth/web/frontend/package.json` — add `jest-axe`,
  `@types/jest-axe`, and `axe-core` as dev dependencies (B7 swap;
  vitest-axe was considered and rejected — see Step 1 rationale).
- Modify: `src/elspeth/web/frontend/vite.config.ts` — register
  `src/test/a11y/setup.ts` in `test.setupFiles` so the
  `toHaveNoViolations` matcher is globally available to the audit
  suite. **Verify exact filename at implementation time** — if the
  project carries a separate `vitest.config.ts`, the setupFiles
  entry belongs there. B7.
- Create: `src/elspeth/web/frontend/src/test/a11y/axe-config.ts` — shared axe configuration.
- Create: `src/elspeth/web/frontend/src/test/a11y/setup.ts` — registers the `toHaveNoViolations` matcher via `expect.extend` (B7).
- Create: `src/elspeth/web/frontend/src/test/a11y/components.a11y.test.tsx` — the audit suite.
- Modify: each component this audit flags (Phases 1-7 additions).

### Setup step

- [ ] **Step 1: Add the axe-core devDependency (B7 swap)**

```bash
cd /home/john/elspeth/src/elspeth/web/frontend
npm install --save-dev jest-axe @types/jest-axe axe-core
```

`jest-axe` provides a `toHaveNoViolations` matcher that's
framework-agnostic — it depends only on `axe-core` (the audit
engine), `chalk`, and `jest-matcher-utils`. It is **not** coupled
to the jest runtime; it works under vitest by registering the
matcher via `expect.extend({ toHaveNoViolations })` in a setup
file (Step 2 below).

`vitest-axe` was considered and rejected (B7): only the stable
`vitest-axe@0.1.0` (Oct 2022, unmaintained) exists for vitest <2;
the `1.0.0-pre.5` pre-release (Jan 2025) pins
`@vitest/pretty-format ^3.0.3` while this project uses vitest
4.1.2 — the dep tree is unsatisfiable. The `jest-axe` +
`expect.extend` pattern is the documented escape hatch and is
used in production by many vitest-based projects.

- [ ] **Step 1.5a: Smoke-test existing suite AFTER install, BEFORE matcher wiring (S3 split, pass 1)**

Immediately after Step 1's `npm install` and BEFORE Step 2's
matcher-registration wiring, verify that adding the dev-dep on its
own does not disturb the existing 525+ vitest tests. At this point
no `expect.extend` has run yet — any regression at this step would
be a node_modules / transitive-dep effect rather than a matcher-
shadowing effect.

```bash
cd src/elspeth/web/frontend && npm test -- --run
```

All existing tests must still pass. If any flake or fail at this
step, document the failure and decide BEFORE proceeding to Step 2
whether to roll back the dev-dep. A regression here is unambiguously
attributable to the install itself (transitive deps, peer-dep
resolution, lockfile drift), not to anything Phase 8 does next.

(Original plan-review concern S9 — vitest-axe adoption changes
test-runtime environment for the entire frontend suite — applies
equally to jest-axe. The S3 split separates "install regressed
something" from "matcher registration regressed something" so the
two failure modes can be diagnosed independently.)

- [ ] **Step 2: Configure axe + mount the matcher (B7)**

Create two files:

**`/home/john/elspeth/src/elspeth/web/frontend/src/test/a11y/axe-config.ts`** —
imports `configureAxe` from `jest-axe` (B7; was `vitest-axe`) and
exports a single configured `axe` instance:

- Disable `color-contrast` (jsdom does not compute CSS-variable
  values, producing false positives; verify contrast manually
  against design tokens).
- `runOnly` with `type: "tag"` and `values: ["wcag2a", "wcag2aa",
  "wcag21a", "wcag21aa"]` (AA only; AAA is too restrictive for a
  developer tool).

**`/home/john/elspeth/src/elspeth/web/frontend/src/test/a11y/setup.ts`** —
at module top, registers the matcher globally. **The module body MUST
begin with this comment block (A6 — register-once invariant seam)**:

```ts
// REGISTER ONCE. This is the project's only global vitest matcher
// registration. Adding additional `expect.extend(...)` calls in test
// files (including audit/a11y test files) will shadow these matchers
// and break the "register once" invariant. If you need a new matcher,
// add it HERE — not in a test file. The audit suite
// (`components.a11y.test.tsx`) deliberately does NOT call
// `expect.extend({ toHaveNoViolations })` for this reason; see Task 7
// Step 4 prose for the same warning at the consumer site.

import { expect } from "vitest";
import { toHaveNoViolations } from "jest-axe";

expect.extend({ toHaveNoViolations });
```

The comment is load-bearing documentation: vitest's `expect.extend`
is idempotent in obvious cases (re-registering the same matcher under
the same name is a no-op) but **NOT** in subtle ones — a future
matcher with a colliding name registered later would silently win,
and the audit suite would start using the wrong implementation. The
comment-as-seam pattern raises the cost of accidental drift.

Register `src/test/a11y/setup.ts` in the vitest config's
`test.setupFiles` array. Verify exact config file at
implementation time: the frontend root carries `vite.config.ts`
which today holds the `test` block; if a separate
`vitest.config.ts` has been split out by then, edit that file
instead. The setupFiles entry runs once per worker, ahead of
every test file — including non-audit tests — which is why Step
1.5a's smoke-test happens *before* this wiring lands.

- [ ] **Step 2.5: Smoke-test existing suite AFTER matcher wiring (S3 split, pass 2)**

Now that the global matcher registration is in place, re-run the
existing suite BEFORE adding the audit suite (Step 4). The
matcher registration runs once per worker for every test file —
including the 525+ pre-existing tests that never asked for axe
behaviour. Any regression at this step is unambiguously
attributable to the `expect.extend({ toHaveNoViolations })` call,
separate from the install-only effect that Step 1.5a already
cleared.

```bash
cd src/elspeth/web/frontend && npm test -- --run
```

All existing tests must still pass. If any flake or fail at this
step (and Step 1.5a was clean), the regression is in the matcher
registration itself — most likely a matcher name collision or a
vitest-runtime peculiarity. Diagnose before adding the audit suite
in Step 4; do **not** add the audit suite on top of a disturbed
baseline.

Rationale for the S3 split: pass-1 (Step 1.5a) catches install-time
breakage; pass-2 (Step 2.5) catches matcher-registration breakage.
Bundling them into a single pre-Step-2 smoke-test (the original
Step 1.5 placement) blurred those two failure modes and gave the
implementer no way to attribute which change caused a regression.

### Audit step

- [ ] **Step 3: Enumerate every new component from Phases 1-7**

Read the file list in this plan's "Sibling plans" section, plus
the manifests in 12-, 13-, 14a-, 14b-, 14c-, 15a-, 15b-, 16a-,
16b-, 16c-, 17-. Produce a list of every `.tsx` component the
phases created. Roughly:

| Phase | Components added |
|---|---|
| 1B | `ComposerPreferencesPanel`, `UserMenu`, `InlineOptOutCheckbox`, `DefaultModeChangedBanner` |
| 2C | `AuditReadinessPanel`, `ReadinessRowDetail`, `ExplainDialog` (B6: real names per `src/elspeth/web/frontend/src/components/audit/`; earlier draft used `AuditReadinessRow`/`AuditReadinessDetail` which do not exist) |
| 3A/3B | `AppHeader`, `HeaderSessionSwitcher`, `HeaderVersionSelector`, `SideRail`, `GraphMiniView`, `YamlExportModal` (B6: `GraphMini` is `GraphMiniView` in tree) |
| 5a | (chat affordances; check 17- for the component list) |
| 6 | (if Phase 6 shipped) `CompletionBar` — a single component with three internal `<button>` elements (Save-for-review, Run-pipeline, Export-YAML). **One axe pass covers all three verbs**; earlier draft enumerating `SaveForReviewButton`/`RunPipelineButton`/`ExportYamlButton` as separate components was wrong (B6: per Phase 6B 19b:38). Note: `ExportYamlButton.tsx` does exist under `components/sidebar/` but is from an unrelated work-stream — list it separately if Phase 3B's side-rail Export-YAML mount is in scope. |
| 7B | reshaped catalog cards + filter chips + search extension |
| 8 | `TutorialReplayButton` (Task 6), refactored `TemplateCards` (Task 3), regrouped `ShortcutsHelp` (Task 5) |

For each component that actually exists in the codebase, write
one axe test.

- [ ] **Step 4: Write the audit suite —
  `components.a11y.test.tsx`** (B7: imports updated)

Create the file with one `describe` per component from the Step 3
enumeration. Each describe has a single `it("has no axe
violations", async () => { ... })` that renders the component with
its minimal required props (mock callbacks as `() => {}`), captures
`container`, and asserts `expect(await
axe(container)).toHaveNoViolations()`.

**(Q12 — audit-suite component-list snapshot test).** Before the
per-component `describe` blocks, add a module-top `describe("audit
surface — coverage snapshot")` containing a single test that
asserts the hard-coded list of audited component names matches a
snapshot:

```ts
const AUDITED_COMPONENTS = [
  "ComposerPreferencesPanel",
  "UserMenu",
  "InlineOptOutCheckbox",
  "DefaultModeChangedBanner",
  "AuditReadinessPanel",
  "ReadinessRowDetail",
  "ExplainDialog",
  "AppHeader",
  "HeaderSessionSwitcher",
  "HeaderVersionSelector",
  "SideRail",
  "GraphMiniView",
  "YamlExportModal",
  "CompletionBar",
  "TutorialReplayButton",
  "TemplateCards",
  "ShortcutsHelp",
  // …populated from Step 3's enumeration; this is illustrative,
  // not exhaustive. The Step 3 list is the source of truth.
] as const;

describe("audit surface — coverage snapshot", () => {
  it("audits exactly the expected component list", () => {
    expect([...AUDITED_COMPONENTS].sort()).toEqual(
      EXPECTED_AUDITED_COMPONENTS_SORTED,
    );
  });
});
```

Where `EXPECTED_AUDITED_COMPONENTS_SORTED` is a hard-coded sorted
copy of the same list, asserted by exact equality. Rationale: a
future PR that deletes a Phase-1-through-7 component (or fails to
add a Phase-9+ component) silently shrinks the audit surface, and
the a11y safety net erodes without anyone noticing. The snapshot
forces every change to the audited set through code review — a
deletion fails the test until the snapshot is intentionally
updated; an addition fails the test until the new component is
named.

The `AUDITED_COMPONENTS` array must also drive the per-component
`describe` blocks (each `describe(name)` block uses a name from
the array), so the array is the single source of truth for both
"what we audit" and "what we claim to audit". A `for (const name
of AUDITED_COMPONENTS)` pattern is acceptable if the project's
vitest convention permits dynamic `describe` generation;
otherwise enumerate the describes by hand and let the snapshot
test guarantee the two stay in sync.

Import `axe` from the configured-axe module created in Step 2:

```ts
import { axe } from "./axe-config";
```

(B7: was `import { axe } from "vitest-axe"`.) Do **not** register
the matcher at module top inside this file — the matcher is
registered globally by `src/test/a11y/setup.ts` (Step 2), wired
into the vitest config's `setupFiles`. Adding a local
`expect.extend({ toHaveNoViolations })` here would shadow the
global registration and break the "register once" invariant the
setup file establishes.

Run: expected mixed (some components pass; some fail).

- [ ] **Step 5: Triage findings**

For each `expect(await axe(...)).toHaveNoViolations()` that fails,
read the violation report. Categorise:

- **High-severity** (blocking, WCAG A or AA): missing labels,
  contrast outside what jsdom can verify but visible in browser,
  missing landmarks, missing focus styles. **Fix in this task.**
- **Medium-severity** (WCAG AAA, or AA but with an acceptable
  workaround): fix opportunistically; if it isn't a 10-minute
  fix, add to `21-phase-9-followups.md`.
- **Low-severity** (axe heuristic flag, e.g. "consider adding aria-current"):
  add to `21-phase-9-followups.md`.

Document each follow-up with the component name, the rule ID
(e.g. `aria-required-attr`), and a single-sentence fix sketch.

- [ ] **Step 6: Fix high-severity findings**

For each high-severity finding, edit the offending component.
Common fix patterns:

- Missing `aria-label` on a `<button>` with only an icon: add it.
- Missing `<label>` paired with an `<input>`: wrap or use `htmlFor`.
- Missing `<h1>`/`<h2>` rhythm: re-check heading nesting.
- Missing `role="status"` / `aria-live` on async feedback regions.
- Modal without `aria-modal="true"` or focus trap: add (project
  already has `useFocusTrap` — use it).

Re-run the test until it passes.

- [ ] **Step 7: Document deferred findings**

If any medium- or low-severity findings remain, create
`/home/john/elspeth/docs/composer/ux-redesign-2026-05/21-phase-9-followups.md`
(if not yet created by Task 4's Case B-1) and append:

```markdown
## Accessibility (axe-core) findings deferred from Phase 8 Task 7

- **HeaderSessionSwitcher**: `aria-current` not set on the active
  session row. Severity: low. Fix: set
  `aria-current="page"` on the row whose id === activeSessionId.
- …
```

- [ ] **Step 8: Run + commit**

`cd src/elspeth/web/frontend && npm test -- --run`. Commit:
`polish(composer): a11y audit + high-severity fixes (Phase 8.7)`.

---

## Task 8: Polish cleanups — dead code, dead comments, lint, CSS consolidation

**Files:**

- Modify: every file in `src/elspeth/web/frontend/src/components/` that
  contains a dead reference, an obsolete comment, or an inconsistent
  CSS variable.
- Modify: every Python file under `src/elspeth/web/` flagged by ruff
  or mypy.

This task is the final sweep. It has no failing-test step because
its tests are the existing test suites — every existing test must
remain green.

### Sweep step

- [ ] **Step 1: Find dead references**

Run `npx ts-prune 2>/dev/null` inside the frontend tree for an
unused-exports report. Cross-check candidates against a hand-built
"no importer" set: `grep -rn '^import.*from' src/` piped to extract
target paths, compared to `find src/ -name '*.ts' -o -name '*.tsx'`.
Inspect each candidate; delete if truly dead. Entry points
(`main.tsx`, `App.tsx`) and test files are expected to survive.

Specifically verify these are gone: old session-sidebar refs (Task
4 should have cleaned them); InspectorPanel refs (Phase 3B deleted
the file); Spec / Runs tab components; the original `TemplateCards`
inline `TEMPLATES` array (Task 3 moved it).

- [ ] **Step 2: Find dead comments**

`grep -rn 'TODO.*phase.[1-7]\|FIXME.*pre-redesign\|XXX.*phase'
src/elspeth/web/ --include='*.py' --include='*.ts' --include='*.tsx'`.
Each match is either a finished work item (delete the comment) or
an unfinished item that should not exist by Phase 8 — surface to
operator.

- [ ] **Step 3: CSS variable consolidation**

Find ad-hoc CSS hex / px / rem values: `grep -rnE 'color:\s*#[0-9a-fA-F]{3,8}|background:\s*#[0-9a-fA-F]{3,8}'
src/elspeth/web/frontend/src/ --include='*.css' | grep -v ':root'`.
For each ad-hoc value, prefer an existing `:root` variable; promote
to a new variable only if the value recurs 3+ times across
components; otherwise leave it (over-tokenisation has cost).

Specifically verify the Phase 8 additions: `.template-card-sda`
(Task 3) uses `var(--color-text-muted)`; `.tutorial-replay-row`
(Task 6) inherits panel spacing tokens; `.shortcuts-group` (Task 5)
heading scale matches other settings sub-headings.

- [ ] **Step 4: Run linters and type-checkers**

Run each in turn; all must exit 0:

- `cd src/elspeth/web/frontend && npm run lint`
- `cd src/elspeth/web/frontend && npx tsc --noEmit`
- `cd src/elspeth/web/frontend && npm run build` — must exit 0
  (Q11). The build step catches a class of breakage that `lint`
  and `tsc --noEmit` miss: production-bundler-only failures from
  CSS module resolution, asset paths, dynamic imports, and the
  Vite-specific path-alias resolution. Type-clean and lint-clean
  code can still fail at `npm run build`; staging deploys hit
  this surface (`project_staging_deployment.md` memory: `npm run
  build` then `systemctl restart elspeth-web.service`), so the
  same command must pass in Phase 8 close.
- `.venv/bin/python -m ruff check src/elspeth/web/`
- `.venv/bin/python -m mypy src/elspeth/web/`

Phase 8 must not check in any lint warnings; if any appear, fix
them in this task.

- [ ] **Step 5: Run the full test suites**

- `.venv/bin/python -m pytest tests/unit/ tests/integration/ -x`
- `cd src/elspeth/web/frontend && npm test -- --run`

All green.

- [ ] **Step 6: Run the tier-model and freeze enforcers**

- `.venv/bin/python scripts/cicd/enforce_tier_model.py check
  --root src/elspeth --allowlist config/cicd/enforce_tier_model`
- `.venv/bin/python scripts/cicd/enforce_freeze_guards.py check
  --root src/elspeth --allowlist config/cicd/enforce_freeze_guards`

Phase 8 should not introduce any new allowlist entries; if it did,
surface to operator.

- [ ] **Step 7: Config-contracts verification + Phase 9 follow-ups cap (S4)**

`.venv/bin/python -m scripts.check_contracts` — green. Phase 8
doesn't touch contracts, but a regression check belongs in the
final sweep.

Then run the Phase 9 follow-ups soft-cap check mandated by
§"Phase 9 follow-ups file — review cadence (S4 — load-bearing)":

```bash
wc -l docs/composer/ux-redesign-2026-05/21-phase-9-followups.md
```

If the line count exceeds **30 lines**, surface to the operator
before Phase 8 close: the follow-ups file has grown beyond the
soft cap at which a Phase 9 scoping read stays cheap. Triage
options the operator can choose between: (a) promote the highest-
severity entries into Filigree issues now so Phase 9 inherits a
smaller file; (b) accept the over-cap file and document the
elevated Phase 9 ingest cost; (c) close Phase 8 with a hard cap
violation noted in the commit message.

Also: if the file's TTL header is still the `[TBD: Phase 8 close
+ 90d]` placeholder from Task 0 Step 4, replace it with the
actual close date (today + 90 days) at this step. The TTL is what
makes the cadence enforceable; leaving the placeholder defeats
the policy.

- [ ] **Step 8: Commit and close out the phase**

Commit: `polish(composer): final sweep — dead code, lint, CSS tokens (Phase 8.8)`.

Final manual verification:

- [ ] Open the composer at `https://elspeth.foundryside.dev` (per
  `project_staging_deployment.md` memory: `npm run build` then
  `systemctl restart elspeth-web.service`).
- [ ] Click through one canonical test case (per
  `project_composer_canonical_test_case.md`): "create a list of 5
  government web pages and use an LLM to rate how cool they are".
- [ ] Verify the empty-state shows the new audit-domain templates.
- [ ] Verify the keyboard shortcuts dialog is grouped.
- [ ] Verify (if Task 6 ran) that the tutorial-replay button
  works end-to-end.
- [ ] Verify (in OTel exporter / metrics endpoint) that the new
  counters surface.
- [ ] **(Q11 — manual contrast verification)** Open the staging
  composer in a browser with the axe DevTools extension installed.
  Run a contrast scan on each of these surfaces and record any
  violations as Phase 9 follow-ups in `21-phase-9-followups.md`
  (do not block Phase 8 close on contrast findings — Task 7's
  jsdom-based axe suite cannot evaluate CSS-variable values and
  this is the compensating manual control):
    - The empty-state (the new audit-domain template cards from
      Task 3).
    - The audit-readiness panel (the Phase 2C surface — text
      colour against the readiness-row backgrounds; severity
      indicators against the row backgrounds).
    - The header session switcher (Task 4 — filter input
      placeholder, archive-button hover/focus states).
    - The settings panel (Task 6 — `TutorialReplayButton`'s
      enabled / disabled / `role="status"` confirmation states).

---

## Risks

| Risk | Severity | Mitigation |
|---|---|---|
| Audit-payload schema-change cohort (B1): operator must delete sessions DB on deploy | **High** | The B1 precondition (record `prior_trust_mode` alongside `trust_mode` in the `trust_mode.changed` audit event) is itself a schema-change cohort: the JSON payload contract changes shape and prior rows lack the new key. The plan calls this out in §"Audit-payload precondition (B1 — load-bearing)" and as a precondition at the top of Task 1 Sub-task 7a so the operator approves both the audit extension and the telemetry wiring together. Mitigated by explicit precondition note + co-land requirement (audit extension and emit in the same commit, or audit extension first as a standalone commit before the emit). |
| Service-signature reshape (B2): test-fixture ripple from Phase 1A | **Medium** | Both `update_composer_preferences` service functions (account-level in `preferences/service.py`; per-session in `sessions/service.py`) change return shape to expose prior + current atomically. Phase 1A established route- and service-level fixtures for both endpoints; those fixtures must update in the same PR as the service-signature change, or unrelated tests will break in CI and be misattributed. Mitigated by explicit precondition note (§"Service signature precondition (B2 — load-bearing)") + "co-land or block" guidance + rejection of the TOCTOU-prone read-before-write alternative. |
| Vocabulary collision between account-level and per-session mode columns (B1-r2) | Resolved | Pass-1 draft shared a single `_ModeName = Literal["guided", "freeform", "unknown"]` across both the account-level helpers (`record_mode_opted_out` / `record_mode_opted_in`) and the per-session helper (`record_session_switched`). The per-session helper receives `trust_mode` values per `src/elspeth/web/sessions/models.py:150` (`'explicit_approve'` / `'auto_commit'`), so every per-session emit would `_assert_mode` → `ValueError`. The W5 swallow wraps *after* the assert (deliberately, so programmer-error guards still crash loudly), so the exception escaped, the audit row had already committed per B1 ordering, and the user PATCH 500ed after the audit row stood — inverting audit primacy at exactly the surface Task 2 instrumented. The fabricated `"unknown"` sentinel was a parallel defect: neither `sessions/models.py:150` (`trust_mode`) nor `sessions/models.py:1076` (`default_composer_mode`) admits NULL or `'unknown'`. Resolution: split the helper module into `_SessionTrustMode = Literal["explicit_approve", "auto_commit"]` (per-session only) and drop the account-level Literal entirely (account-level helpers are post-state-only per B2.b and take no mode kwarg). Regression test asserts `from_mode="guided"` (cross-vocabulary leak) raises `ValueError`. See §"Vocabulary discipline (B1-r2 — load-bearing)" inside §"Account-level scope narrowing (B2.b — load-bearing)". |
| Account-level telemetry loses per-from-state breakdown (B2.b) | **Low** | Design doc 10 §Phase 8 wants "opt-out rate" (a ratio), still measurable as `composer.mode.opted_out_total / composer.preferences.patch_total{mode_changed=True}` (see §W4 / B3-r3 for the denominator-correction reasoning — bare `patch_total` is deflated by banner-only PATCHes and the ratio is a set-rate on the `default_mode` field, not a transition-rate). What is lost is per-from-state attribution — Phase 8 cannot answer "of users who opted out, how many were on `guided` vs `unknown`?" Acceptable; preserves the prior architectural decision in the "Operational signal only" module-level comment in `preferences/service.py` (preferences are operational-only, no Landscape emit). If the per-from-state breakdown is needed later, it requires promoting account-level preferences to audit — explicitly rejected for Phase 8 in §"Account-level scope narrowing (B2.b — load-bearing)" because (a) it overrides a deliberate prior architectural decision; (b) it is a Tier-1 schema change subject to the "DB migration = delete the old DB" policy; (c) the design doc never asked for it. |
| Phase 8 probe-failure counter bootstrap-order coupling (W8-r2) | Resolved | Pass-1 draft added `phase_8_probe_failed` as a slot on the `frozen=True, slots=True` `_SessionsTelemetry` dataclass. That required Task 1 Step 5 (container extension) to land before Task 0 Step 2 could emit, or AttributeError on the slots dataclass — a hidden cross-task precondition exposed as either "complete Step 5 first" or a bootstrap-only Task 1 Step 0. Naming was also inconsistent: the field `phase_8_probe_failed` violated the plan's own `_total` suffix convention used by every other counter (Q3/S7). Resolution (W8-r2 module-local counter): move the counter out of the container entirely. Declare `_PHASE_8_PROBE_FAILED_COUNTER = _meter.create_counter("composer.phase_8.probe_failed_total", ...)` at module-import time in `telemetry_phase8.py`. Matches the existing `_PREFERENCES_PATCH_COUNTER` pattern in `src/elspeth/web/preferences/service.py:85`. Task 0 is now structurally self-contained: it can run in any order relative to Task 1 Step 5. The `_total` suffix is now uniform across all Phase 8 counters. See Task 0 Step 2, Task 1 Step 4 module shape, and Task 1 Step 5's deliberately-absent comment for cross-references. |
| Phase 6 mid-delivery — Sub-task 7d probe (B3 cohort a) may miss if Phase 6 lands AFTER Phase 8 starts | **Medium** | Coordinate Phase 6 close + Phase 8 start ordering. If Phase 6 ships first, Task 0's probe hits Case A and Sub-task 7d wires the two `composer.share.*` counter emits at Phase 6's verify-failure and expiry-hit branches. If Phase 6 lands during or after Phase 8, Sub-task 7d is a documented no-op (counters stay in the container, unused) and the wiring is filed as a Phase 9 follow-up — **operator surface required** per §"Probe safety policy" so the timing collision is visible at close. The original "minimum instrumentation bar" framing that asserted Phase 6 would emit these was incorrect (B3 finding); Phase 8 explicitly owns the emits now. See §"Cross-phase telemetry — cohort split (B3 reshape)". |
| Telemetry reveals the default-guided call was wrong (opt-out rate is high) | **Acceptable** — design doc 10 §Risk table explicitly names this as acceptable. Telemetry's job is to surface this. | None. If opt-out rate is high, that's signal, not failure. The product team adjusts; the redesign is *informed by data*, not pinned to a guess. |
| Task 4 surfaces that Phase 3B descoped HeaderSessionSwitcher | Medium | Probe in Step 1 catches this; surfaces to operator rather than absorbing scope. |
| Task 6 surfaces that Phase 4 didn't ship the flag | Medium | Probe in Step 1 catches this; defers to Phase 9. |
| Task 7's axe audit floods with findings that block the phase | Medium | Triage rule: only high-severity fixed inline; medium/low recorded as Phase 9. The audit is a *measurement*, not an absolute gate. |
| Templates → README mapping (Task 3) gets out of sync with README later | Low | `templates_data.test.ts` Step 2 snapshot test catches this if README content changes; CI fails until the test is updated alongside the README. |
| Shortcuts reorganisation (Task 5) breaks muscle memory | Low | Documented in commit message; the ? help dialog stays in place; no shortcut binding actually changes in Task 5 — only the *display* groups them. |
| CSS variable consolidation (Task 8 Step 3) changes a visible style | Low | Manual verification step exercises the canonical test case; any visual regression surfaces immediately. |
| Tutorial-replay counter semantics (was: replay counter fires on every click without re-completing the tutorial) | **Resolved** — counter deferred to Phase 9 per Decision 2 / Option C. The button still ships in 8c (UI-only); the boundary question for the counter (user-write-intent vs CLAUDE.md superset exception) is open in `21-phase-9-followups.md`. Phase 9 may answer it by accepting the counter as telemetry-only with an audit row, by broadening the superset exception project-wide, or by dropping the counter entirely. |
| `jest-axe` + `axe-core` add 5+ MB to `node_modules` (B7: was vitest-axe) | Low | Dev dependency only; bundle is unchanged. |
| W5 try/except wrap establishes a new project-wide convention applied only to new Phase 8 emits (A7) | Medium | Existing emits (e.g., `_PREFERENCES_PATCH_COUNTER.add` in `src/elspeth/web/preferences/service.py`) still call `.add()` bare. Phase-8-added routes cannot 500 from a broken OTel exporter; pre-Phase-8 routes can. File as a separate Filigree ticket: "apply W5 try/except pattern to all existing OTel counter sites in `src/elspeth/web/`" to lift this from a per-phase convention to a project-wide one. Without that backfill, the audit-primacy guarantee Phase 8 establishes for its emit sites does not hold for pre-existing emit sites — a subtle invariant gap that would silently re-invert primacy on any pre-Phase-8 PATCH whose OTel exporter happened to fail. |
| Original axe-matcher choice (`vitest-axe`) was incompatible with vitest 4.1.2 (B7) | Resolved | Switched to `jest-axe@10.0.0` mounted via `expect.extend({ toHaveNoViolations })` in `src/test/a11y/setup.ts`, registered through the vitest config's `test.setupFiles`. `jest-axe` has no jest runtime coupling; it depends only on `axe-core`, `chalk`, and `jest-matcher-utils`, and works under vitest. Rationale: `vitest-axe`'s stable `0.1.0` (Oct 2022, unmaintained) targets vitest <2; the `1.0.0-pre.5` pre-release pins `@vitest/pretty-format ^3.0.3` while the project uses vitest 4.1.2 — the dep tree is unsatisfiable. The cohort risk (global matcher registration disturbing the existing 525+ vitest suite — original S9) is mitigated by Task 7 Steps 1.5a + 2.5 (split smoke-test per S3: 1.5a post-install / 2.5 post-matcher-wiring). |
| Dead-code sweep (Task 8) catches accumulated drift from 7 phases; future phases will leave their own drift | Low | After Phase 8 ships, add a CI lint rule that fails on exported-but-unused constants (`ts-prune` or equivalent). Future phases should delete dead exports in the same commit that removes their dispatch — Phase 8 is a one-time archaeology task, not a repeating pattern. |

---

## Review history

Extracted (2026-05-19, Q13 resolution) to a sibling file:
[`20-phase-8-review-history.md`](20-phase-8-review-history.md). That
file contains the full pass-by-pass trail (2026-05-15 panel,
2026-05-19 B3 reshape, 2026-05-19 pass-2 findings, 2026-05-19 pass-3
findings + Decision 1/Decision 2 resolutions). The `(Bn-rN — load-
bearing)` references in the plan body point back to those entries
for context.

Future review passes append to the sibling file, not to this plan.
