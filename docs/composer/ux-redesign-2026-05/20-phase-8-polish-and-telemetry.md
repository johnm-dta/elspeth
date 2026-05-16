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
- **Phase 4** — Hello-world tutorial (plan not yet written); Task 6 conditional on `tutorial_completed` flag.
- **Phase 5a** ([17-phase-5a-dynamic-source-from-chat.md](17-phase-5a-dynamic-source-from-chat.md)) / **Phase 5b** — Task 1 wires Phase 5a marker if present.
- **Phase 6** — Completion gestures (plan not yet written); Task 1 wires YAML-export marker if present.
- [16a/b/c — Phase 7](16a-phase-7a-backend.md) — catalog reshape; Task 3 follows its vocabulary and tone.

**Roadmap reference:** [00-implementation-roadmap.md](00-implementation-roadmap.md).
**Design reference:** [10-implementation-phasing.md](10-implementation-phasing.md) §"Phase 8".

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
  clears `composer.tutorial_completed` (Task 6). **Conditional** on
  Phase 4 having shipped the flag.
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

- `composer.tutorial_completed` flag (Task 6): crash-on-anomaly per
  [12-phase-1a-backend.md](12-phase-1a-backend.md). Corrupt (non-boolean) value → 500.

**Tier 3 — External data (source input):**

- **Tutorial-replay PATCH body**: Pydantic `Literal[False]` / `bool`
  rejects non-boolean at the boundary (422). `"false"` (str) ≠ `False` (bool).
- **README "Example Use Cases" content** (Task 3): build-time only —
  hand-curated into `TemplateCards.tsx`; no runtime Tier 3 boundary.

### Telemetry primacy explicit acknowledgment

Per [CLAUDE.md](/home/john/elspeth/CLAUDE.md) §"Telemetry and Logging"
the order is **audit first** (sync, crash-on-failure), **telemetry
next** (async, best-effort), **logging last** (only when audit and
telemetry are broken). Phase 8 adds telemetry **only** — no new audit
events, no new logging.

What this phase sends to the OTel meter:

- `composer.mode.opted_out_total{from_mode}` — counter, incremented
  inside the PATCH `/api/composer-preferences` route whenever a user's
  mode transitions to `freeform` (Task 2). Aggregate. No user ID, no
  session ID. Attribute set: `{from_mode: "guided" | "unknown"}`.
- `composer.mode.opted_in_total` — counter, the symmetric case
  (someone re-selecting guided after having freeform). Same attribute
  shape.
- `composer.session.completed_total{mode, completion_verb}` — counter,
  fired by the completion-gestures emit-site in Phase 6 (Task 2 wires
  this if Phase 6 shipped). Attributes: `{mode: "guided" | "freeform",
  completion_verb: "save_for_review" | "run_pipeline" | "export_yaml"}`.
- `composer.session.switched_total{from_mode, to_mode}` — counter,
  fired whenever a user explicitly switches a session's mode mid-flow
  (Task 2). Attributes: `{from_mode, to_mode}`.
- `composer.tutorial.started_total` — counter (Task 6, conditional).
- `composer.tutorial.completed_total` — counter (Task 6, conditional).
- `composer.tutorial.replayed_total` — counter, fired by the new
  replay button (Task 6).
- `phase_8_probe_failed{phase, probe}` — counter, fired by Task 0's
  probe-mechanism checks when an upstream phase probe returns
  "not found." Signals a conditional task that cannot run; ensures
  the absence is recorded rather than silently producing degraded
  metrics (see §Verification approach — Probe safety policy).

What this phase **does not** send: prompt text, session IDs, user IDs,
audit-trail events, or any structlog/logger calls (zero new logging
statements; counter-emit failures surface via OTel exporter logs only).

The **superset rule** holds: every operational signal Phase 8 emits is
already represented in the audit trail (session events, preference
PATCH events). Telemetry is a strict subset of audit-recorded reality.

**Channel decision — no frontend telemetry primitive in Phase 8.** Every
counter listed above emits from a backend handler (the `composer-preferences`
PATCH route, the session-mutation routes, the completion-gesture routes
that Phase 6 already audits). Phase 3A's and Phase 1B's deferred
"frontend telemetry breadcrumb" notes (see
[15a1-phase-3a-removals-part-1.md](15a1-phase-3a-removals-part-1.md)
lines 1380-1382 and 1442, and the breadcrumb deferral in
[15a2-phase-3a-removals-part-2.md](15a2-phase-3a-removals-part-2.md)
line 690) are resolved **by emitting from the backend route the
frontend action invokes**, not by adding a frontend telemetry module.
Rationale: the backend already audits the action; emitting an OTel
counter at the same site preserves audit primacy without introducing a
second emission channel that could drift from the audit record (the
superset rule above). If a future operational signal genuinely requires
frontend-only emission (e.g., a UI interaction the backend never sees),
that decision belongs to a successor phase with its own primacy
analysis; Phase 8 does not establish a frontend telemetry primitive.

---

## Minimum instrumentation bar for prior phases

The following metrics should have been emitted by earlier phases.
Task 0 verifies their presence; absent metrics produce a Phase 9
follow-up and are **not** backfilled by Phase 8.

| Phase | Expected metric | Signal |
|-------|----------------|--------|
| 2 | `composer.audit.render_duration` | Audit panel render latency |
| 2 | `composer.audit.fetch_failure_total` | Audit panel fetch failure rate |
| 5b | `composer.interpretation.resolve_duration` | Interpretation resolve latency |
| 5b | `composer.interpretation.opt_out_total` | Interpretation opt-out rate |
| 6 | `composer.share.token_verify_failure_total` | Shareable-link token verify failure rate |
| 6 | `composer.share.link_expiry_hit_total` | Shareable-link expiry hit rate |

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
src/elspeth/web/frontend/src/test/a11y/components.a11y.test.tsx
docs/composer/ux-redesign-2026-05/21-phase-9-followups.md     (only if any follow-ups accumulate)
```

**Files this plan modifies:**

- `src/elspeth/web/sessions/routes.py` (Task 1: telemetry emit on PATCH)
- `src/elspeth/web/sessions/service.py` (Task 1, 2: counter helpers)
- `src/elspeth/web/sessions/telemetry.py` (Task 2: new counters in container)
- `src/elspeth/web/composer/service.py` (Task 1: wire deferred markers)
- `src/elspeth/web/frontend/src/components/chat/TemplateCards.tsx`,
  `.test.tsx` (Task 3)
- `src/elspeth/web/frontend/src/components/sessions/SessionSidebar.tsx`
  (Task 4: deletion)
- `src/elspeth/web/frontend/src/components/header/HeaderSessionSwitcher.tsx`
  (Task 4: filter, archive)
- `src/elspeth/web/frontend/src/App.tsx` (Task 5: regroup shortcuts;
  Task 4: drop sidebar mount)
- `src/elspeth/web/frontend/src/components/common/ShortcutsHelp.tsx`,
  `.test.tsx` (Task 5)
- `src/elspeth/web/frontend/src/components/settings/ComposerPreferencesPanel.tsx`
  (Task 6: mount replay button)
- `src/elspeth/web/frontend/src/api/preferences.ts` (Task 6)
- `src/elspeth/web/frontend/src/state/preferencesStore.ts` (Task 6)
- `src/elspeth/web/frontend/package.json` (Task 7: vitest-axe dev dep)
- `src/elspeth/web/frontend/src/components/**/*.tsx` (Task 7, 8: a11y
  + polish across new components)

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

### Probe safety policy

Probes in conditional tasks check whether an upstream phase shipped
its surface. **If a probe returns "not found":**

- The conditional task is a documented no-op (record in
  `21-phase-9-followups.md`).
- The metric is **not** emitted with degraded/inferred values.
  Partial telemetry is worse than no telemetry — fail safe.
- Emit `phase_8_probe_failed{phase=X, probe=Y}` via the OTel meter
  so the absence is visible in dashboards. This counter signals
  "this conditional task could not run," not an error.

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
| Task 4 | `ls ...HeaderSessionSwitcher.tsx` | file exists |
| Task 6 | `grep -rn 'tutorial_completed'` in web/ | ≥1 match |

- [ ] **Step 2: For each missed probe, emit the probe-failure counter**

```python
# In telemetry_phase8.py (added after Step 4 of Task 1):
phase_8_probe_failed.add(1, attributes={"phase": "X", "probe": "Y"})
```

Record in `21-phase-9-followups.md` for each miss.

- [ ] **Step 3: Confirm no conditional task proceeds on a missed probe**

Review each task whose probe returned a miss. Confirm the plan
routes it to Case B-1/B-2 (documented no-op or operator surface).
No task may emit a metric whose upstream signal was never recorded.

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
counter increments. Required test cases:

- `record_mode_opted_out(tel, from_mode="guided")` increments
  `tel.mode_opted_out_total` by 1; the recorded attribute dict is
  exactly `{"from_mode": "guided"}` (read via the `calls` attribute
  after type-narrowing to `_FakeCounter`).
- `record_mode_opted_in(tel, from_mode="freeform")` increments
  `tel.mode_opted_in_total` by 1 with `{"from_mode": "freeform"}`.
- `record_session_switched(tel, from_mode="guided",
  to_mode="freeform")` records `{"from_mode": "guided", "to_mode":
  "freeform"}` on `tel.session_switched_total`.
- Parametrise the from_mode argument over `{"guided", "freeform",
  "unknown"}` and assert each accepts.
- Pass `from_mode="kiosk"` (not in the Literal set) and assert
  `ValueError` matching `"from_mode must be"`. This is the offensive
  runtime guard; the helper crashes loudly rather than silently
  recording garbage.

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

- Re-export the container as `SessionsTelemetry` (avoid leaking the
  underscore name across the composer surface).
- `_ModeName = Literal["guided", "freeform", "unknown"]` and a
  matching frozenset `_KNOWN_MODES` for the runtime check.
- `_assert_mode(name, value)` — raises `ValueError` with
  message `f"{name} must be one of {sorted(_KNOWN_MODES)!r}; got
  {value!r}"` when value is outside the literal set.
- `record_mode_opted_out(tel, *, from_mode)` — calls
  `tel.mode_opted_out_total.add(1, attributes={"from_mode":
  from_mode})` after the assert.
- `record_mode_opted_in(tel, *, from_mode)` — symmetric on
  `tel.mode_opted_in_total`.
- `record_session_switched(tel, *, from_mode, to_mode)` — asserts
  both modes, then `tel.session_switched_total.add(1,
  attributes={"from_mode": ..., "to_mode": ...})`.

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
tutorial_replayed_total: _Counter
session_completed_total: _Counter
```

And in `build_sessions_telemetry()` extend both branches (`if meter
is None` and the real-meter branch). Counter names in the
real-meter branch:

- `composer.mode.opted_out_total`
- `composer.mode.opted_in_total`
- `composer.session.switched_total`
- `composer.tutorial.started_total`
- `composer.tutorial.completed_total`
- `composer.tutorial.replayed_total`
- `composer.session.completed_total`

- [ ] **Step 6: Run the test**

`.venv/bin/python -m pytest src/elspeth/web/composer/telemetry_phase8_test.py -v` — all tests pass.

- [ ] **Step 7: Wire deferred markers (per Step 1's enumeration)**

For each marker location found in Step 1, replace the marker comment
with the appropriate emit call from `telemetry_phase8.py`.

**Sub-task 7a (always required — Phase 1A's PATCH route):**

In `update_composer_preferences` in `routes.py`, after the audit
event persists and before the response is constructed, branch on
the `old_record.trust_mode → prefs.trust_mode` transition: if the
new mode is `freeform`, call `record_mode_opted_out(telemetry,
from_mode=old_record.trust_mode)`; if `guided`, call
`record_mode_opted_in(...)`. Both are no-ops when the modes are
equal. Audit primacy: the audit event is already written when this
emit runs; a telemetry failure here does not invalidate the audit
record.

**Sub-task 7b (conditional — Phase 5a dynamic-source emit):**

Probe: `grep -n 'dynamic.source\|dynamic_source' src/elspeth/web/composer/service.py | head -10`

If the dynamic-source-from-chat path exists from Phase 5a but the
emit is missing, add a new counter `composer.source.dynamic_created_total`
to the container (Step 5) and emit it at the dynamic-source creation
site. If the path doesn't exist (Phase 5a didn't ship), record
this as a Phase 9 follow-up and continue.

**Sub-task 7c (conditional — Phase 6 YAML-export event):**

Probe: `grep -rn 'export_yaml\|Export YAML' src/elspeth/web/frontend/src/ | head -10`

If a YAML-export completion verb exists (Phase 6 landed), wire
`composer.session.completed_total{completion_verb="export_yaml"}`
at the click handler's network-success branch. If Phase 6 didn't
ship, the counter exists in the container (added in Step 5) but
nothing emits to it; record as Phase 9 follow-up.

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

Run: expected fail (no emit at route level yet).

- [ ] **Step 3: Implement the route emit**

In `update_composer_preferences` in `routes.py`, after the audit
event persists and **before** the response is constructed:

```python
if old_record.trust_mode != prefs.trust_mode:
    record_session_switched(
        telemetry,
        from_mode=old_record.trust_mode,
        to_mode=prefs.trust_mode,
    )
```

Note this runs **alongside** the opt-out/opt-in emit from Task 1.
The two are complementary: opt-out/opt-in is the user-level
"defaulting away from guided" signal; session-switched is the
session-level "this session changed mode" signal. Both increment.

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

- Read: `src/elspeth/web/frontend/src/components/header/HeaderSessionSwitcher.tsx` (existence probe).
- Modify (if it exists): `HeaderSessionSwitcher.tsx` — add filter + archive controls.
- Modify (if migration is ready): `src/elspeth/web/frontend/src/App.tsx` — drop the `SessionSidebar` mount.
- Delete (if migration is ready):
  - `src/elspeth/web/frontend/src/components/sessions/SessionSidebar.tsx`
  - `src/elspeth/web/frontend/src/components/sessions/SessionSidebar.test.tsx`

### Probe step

- [ ] **Step 1: Probe — does HeaderSessionSwitcher exist?**

```bash
ls /home/john/elspeth/src/elspeth/web/frontend/src/components/header/HeaderSessionSwitcher.tsx 2>/dev/null
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
cat /home/john/elspeth/src/elspeth/web/frontend/src/components/header/HeaderSessionSwitcher.tsx
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

- Probe: `grep -n 'archiveSession\|archived' src/elspeth/web/frontend/src/state/sessionStore.ts`.
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

- Read + modify: `src/elspeth/web/frontend/src/App.tsx` (lines
  86-179 in the current head; the keyboard handler block).
- Modify: `src/elspeth/web/frontend/src/components/common/ShortcutsHelp.tsx`
- Modify: `src/elspeth/web/frontend/src/components/common/ShortcutsHelp.test.tsx`

### Inventory step

- [ ] **Step 1: Inventory existing shortcuts**

The current `handleKeyDown` (App.tsx:86-179) defines:

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

If Step 2 found Alt+1-4 dead post-Phase-3, delete that block from
`handleKeyDown` (App.tsx:124-139). Same for Ctrl+Shift+V if its
consumer is gone.

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
- Modify: `src/elspeth/web/frontend/src/api/preferences.ts` (add `clearTutorialCompleted()`).
- Modify: `src/elspeth/web/frontend/src/state/preferencesStore.ts` (expose `replayTutorial()`).
- Modify: `src/elspeth/web/composer/telemetry_phase8.py` (add the replay helper).

### Probe step

- [ ] **Step 1: Probe — does the tutorial flag exist?**

```bash
grep -rn 'tutorial_completed\|tutorialCompleted\|composer.tutorial' \
  /home/john/elspeth/src/elspeth/web/ --include='*.py' --include='*.ts' --include='*.tsx'
```

**Case A — flag exists (Phase 4 shipped):** continue with Steps 2-8.

**Case B — flag does not exist:** Phase 4 has not shipped the
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

Phase 4 should have added `tutorial_completed: bool` to the
`user_preferences` row and made it readable / writeable on the
PATCH /GET endpoint. Confirm:

```bash
grep -n 'tutorial_completed' /home/john/elspeth/src/elspeth/web/sessions/routes.py \
  /home/john/elspeth/src/elspeth/web/sessions/service.py
```

If the field is there, the replay action is `PATCH
/api/composer-preferences` with body `{"tutorial_completed":
false}`. No new endpoint needed.

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

Run: expected fail.

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
`api.updateComposerPreferences({ tutorial_completed: false })` then
re-invokes `get().bootstrap()` so the cached row reflects the
freshly-cleared flag without a race.

In `api/preferences.ts`, verify `updateComposerPreferences`'s body
type already includes `tutorial_completed` (Phase 4 should have
extended it; if not, extend it here).

- [ ] **Step 6: Mount the button in `ComposerPreferencesPanel`**

Add near the bottom of the panel's JSX a `<section
aria-labelledby="tutorial-section-heading">` with an `<h3
id="tutorial-section-heading">Tutorial</h3>` and the
`<TutorialReplayButton />` mount.

- [ ] **Step 7: Emit the replay counter**

In the backend PATCH handler for `composer-preferences`, detect a
transition where `tutorial_completed` changes from `True` to
`False` (per-field comparison against `old_record`) and emit
`tel.tutorial_replayed_total.add(1)`. Or extend
`telemetry_phase8.py` with a `record_tutorial_replayed(tel)`
helper for symmetry with the other emit helpers. The counter
already exists in the container from Task 1 Step 5.

- [ ] **Step 8: Run tests + commit**

`cd src/elspeth/web/frontend && npm test -- --run` and
`.venv/bin/python -m pytest src/elspeth/web/sessions/
src/elspeth/web/composer/`. Commit: `feat(composer): add
tutorial-replay affordance in settings (Phase 8.6)`.

---

## Task 7: Accessibility audit (axe-core)

**Files:**

- Modify: `src/elspeth/web/frontend/package.json` — add `vitest-axe`
  and `axe-core` as dev dependencies.
- Create: `src/elspeth/web/frontend/src/test/a11y/axe-config.ts` — shared axe configuration.
- Create: `src/elspeth/web/frontend/src/test/a11y/components.a11y.test.tsx` — the audit suite.
- Modify: each component this audit flags (Phases 1-7 additions).

### Setup step

- [ ] **Step 1: Add the axe-core devDependency**

```bash
cd /home/john/elspeth/src/elspeth/web/frontend
npm install --save-dev vitest-axe axe-core
```

`vitest-axe` provides a `toHaveNoViolations` matcher for
`expect()`; it wraps `axe-core` (the audit engine).

- [ ] **Step 2: Configure axe**

Create `/home/john/elspeth/src/elspeth/web/frontend/src/test/a11y/axe-config.ts`.
The module imports `configureAxe` from `vitest-axe` and exports a
single `axe` instance configured to:

- Disable `color-contrast` (jsdom does not compute CSS-variable
  values, producing false positives; verify contrast manually
  against design tokens).
- `runOnly` with `type: "tag"` and `values: ["wcag2a", "wcag2aa",
  "wcag21a", "wcag21aa"]` (AA only; AAA is too restrictive for a
  developer tool).

### Audit step

- [ ] **Step 3: Enumerate every new component from Phases 1-7**

Read the file list in this plan's "Sibling plans" section, plus
the manifests in 12-, 13-, 14a-, 14b-, 14c-, 15a-, 15b-, 16a-,
16b-, 16c-, 17-. Produce a list of every `.tsx` component the
phases created. Roughly:

| Phase | Components added |
|---|---|
| 1B | `ComposerPreferencesPanel`, `UserMenu`, `InlineOptOutCheckbox`, `DefaultModeChangedBanner` |
| 2C | `AuditReadinessPanel`, `AuditReadinessRow`, `AuditReadinessDetail` |
| 3A/3B | `AppHeader`, `HeaderSessionSwitcher`, `HeaderVersionSelector`, `SideRail`, `GraphMini`, `YamlExportModal` |
| 5a | (chat affordances; check 17- for the component list) |
| 6 | `CompletionBar`, `SaveForReviewButton`, `RunPipelineButton`, `ExportYamlButton` (if Phase 6 shipped) |
| 7B | reshaped catalog cards + filter chips + search extension |
| 8 | `TutorialReplayButton` (Task 6), refactored `TemplateCards` (Task 3), regrouped `ShortcutsHelp` (Task 5) |

For each component that actually exists in the codebase, write
one axe test.

- [ ] **Step 4: Write the audit suite —
  `components.a11y.test.tsx`**

Create the file with one `describe` per component from the Step 3
enumeration. Each describe has a single `it("has no axe
violations", async () => { ... })` that renders the component with
its minimal required props (mock callbacks as `() => {}`), captures
`container`, and asserts `expect(await
axe(container)).toHaveNoViolations()`. At module top, register the
matcher via `expect.extend({ toHaveNoViolations })`.

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

- [ ] **Step 7: Config-contracts verification**

`.venv/bin/python -m scripts.check_contracts` — green. Phase 8
doesn't touch contracts, but a regression check belongs in the
final sweep.

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

---

## Risks

| Risk | Severity | Mitigation |
|---|---|---|
| Telemetry reveals the default-guided call was wrong (opt-out rate is high) | **Acceptable** — design doc 10 §Risk table explicitly names this as acceptable. Telemetry's job is to surface this. | None. If opt-out rate is high, that's signal, not failure. The product team adjusts; the redesign is *informed by data*, not pinned to a guess. |
| Task 4 surfaces that Phase 3B descoped HeaderSessionSwitcher | Medium | Probe in Step 1 catches this; surfaces to operator rather than absorbing scope. |
| Task 6 surfaces that Phase 4 didn't ship the flag | Medium | Probe in Step 1 catches this; defers to Phase 9. |
| Task 7's axe audit floods with findings that block the phase | Medium | Triage rule: only high-severity fixed inline; medium/low recorded as Phase 9. The audit is a *measurement*, not an absolute gate. |
| Templates → README mapping (Task 3) gets out of sync with README later | Low | `templates_data.test.ts` Step 2 snapshot test catches this if README content changes; CI fails until the test is updated alongside the README. |
| Shortcuts reorganisation (Task 5) breaks muscle memory | Low | Documented in commit message; the ? help dialog stays in place; no shortcut binding actually changes in Task 5 — only the *display* groups them. |
| CSS variable consolidation (Task 8 Step 3) changes a visible style | Low | Manual verification step exercises the canonical test case; any visual regression surfaces immediately. |
| Tutorial-replay button (Task 6) fires the replay counter without the user completing the tutorial again | Acceptable | The counter measures "user intent to replay," not "user completed tutorial again." The completion counter (`tutorial_completed_total`) is separate and fires from Phase 4's completion event. |
| `vitest-axe` adds 5+ MB to `node_modules` | Low | Dev dependency only; bundle is unchanged. |
| Dead-code sweep (Task 8) catches accumulated drift from 7 phases; future phases will leave their own drift | Low | After Phase 8 ships, add a CI lint rule that fails on exported-but-unused constants (`ts-prune` or equivalent). Future phases should delete dead exports in the same commit that removes their dispatch — Phase 8 is a one-time archaeology task, not a repeating pattern. |

---

## Review history

**2026-05-15 — review-panel findings applied (5 items)**

- **F1 (Critical/Systems):** Partial-telemetry signals can mislead. Added Task 0
  (Probe-mechanism tests) before Task 1 to verify each upstream phase probe returns
  the expected signal before conditional branches run. Added a fail-safe policy to
  §Verification approach: conditional tasks whose probe returns "phase not shipped"
  must emit `phase_8_probe_failed{phase, probe}` via telemetry rather than silently
  emitting degraded metrics. Added `phase_8_probe_failed` to the OTel counter list.
- **F2 (Important/Architecture+Systems):** "WE HAVE NO USERS YET" tension. Added
  §Caretaker logic policy explaining why caretaker logic is currently nullified by
  the delete-the-DB policy and that Phase 8 wires it correctly so Phase 9's
  migration-runner gives it permanent meaning.
- **F3 (Important/Architecture):** Probe mechanism not tested. Resolved by F1 above.
- **F4 (Important/Quality):** Some telemetry belongs inline with earlier phases.
  Added §Minimum instrumentation bar listing the metrics that should have been
  emitted by Phases 2, 5b, 6. Task 0 verifies their presence; absent metrics
  produce a Phase 9 follow-up rather than a Phase 8 backfill.
- **F5 (Suggestion/Architecture):** Dead-code sweep at end of Phase 8 is archaeology.
  Added a Risks row recommending a CI lint rule on exported-but-unused constants
  post-Phase-8 and noting that future phases should clean up after themselves.
