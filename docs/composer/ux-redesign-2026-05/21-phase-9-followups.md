# Phase 9 follow-ups — items deferred from Phase 8

> **Created:** 2026-05-19 (Phase 8 pre-flight gate).
> **Review by:** Phase 8 close + 90 days. If Phase 9 has not begun by
> that date, each remaining entry must either be filed as a Filigree
> issue, promoted into a Phase 9 plan section, or explicitly retired
> with a rationale. Leaving entries in this file unreviewed past the
> TTL recapitulates the failure mode documented in the project memory
> `project_cicd_allowlist_audit_2026-05-19.md` (51% growth in 32 days
> on the L3 allowlist because no expiry mechanism forced periodic
> review).
> **Cadence reference:** [20-phase-8-polish-and-telemetry.md](20-phase-8-polish-and-telemetry.md)
> §"Phase 9 follow-ups file — review cadence (S4 — load-bearing)".

This file is the durable record of every item Phase 8 explicitly
deferred or de-scoped. Entries fall into four classes; each entry
states which class, what triggered the deferral, and what closes it.

| Class | Definition | Closure path |
|---|---|---|
| **Probe-miss no-op** | A conditional Phase 8 sub-task whose upstream-phase probe returned "not found" at the 2026-05-19 gate. The sub-task did not run; the gap is documented here. | Upstream phase ships the surface, then file a Filigree issue under Phase 9 to wire the deferred sub-task. |
| **Decision deferral** | A boundary question Phase 8 chose not to resolve (Decision-2 resolution = Option C). | Phase 9 design pass resolves the boundary question; do not silently re-categorise the deferred surface. |
| **Performance instrumentation** | B3 cohort (c) — perf signals from already-shipped phases. Reopening shipped phases for pure perf instrumentation was out of proportion to the benefit. | Phase 9 perf-instrumentation pass adds these counters with the original phase's vocabulary. |
| **A11y medium/low finding** | Task 7 axe-core findings classified medium/low severity at audit time. High-severity findings were fixed inline; medium/low were recorded here. | Phase 9 a11y closeout sweep, or earlier if Filigree-promoted. |

---

## Pre-flight gate probe outcomes (recorded 2026-05-19)

The Phase 8 pre-flight (W1 — operator gate) ran six probes against
the `feat/composer-phase-8-polish` worktree branched from RC5.2
(`dd20888f0`). Outcomes:

| Probe | Hits | Gated sub-task | Class | Class-A or Class-B disposition |
|---|---|---|---|---|
| `grep -rn 'telemetry: deferred to Phase 8' src/elspeth/` | 0 | Task 1 Sub-task 7a markers-harvest framing | Probe-miss no-op | Class B — scope OUT of Phase 8 |
| `grep -rn 'verify_token\|verify_share_token' src/elspeth/web/` | 0 | Task 1 Sub-task 7d (B3 cohort a — Phase 6 verify-failure / expiry-hit emits) | Probe-miss no-op | Class B — scope OUT of Phase 8 |
| `grep -rn 'auto_interpreted_opt_out' src/elspeth/web/` | ≥1 | Task 1 Sub-task 7e (B3 cohort b1 — Phase 5b interpretation-opt-out emit) | (HIT — scoped IN) | n/a |
| `grep -rn 'audit_readiness\|composer/audit/readiness' src/elspeth/web/` | ≥1 | Task 1 Sub-task 7f (B3 cohort b2 — Phase 2C audit-fetch-failure emit) | (HIT — scoped IN) | n/a |
| `ls src/elspeth/web/frontend/src/components/sessions/HeaderSessionSwitcher.tsx` | file exists | Task 4 (session-sidebar migration) | (HIT — scoped IN) | n/a |
| `grep -rn 'tutorial_completed' src/elspeth/web/` | 0 | Task 6 (tutorial-replay button + counter) | Probe-miss no-op | Class B — scope OUT of Phase 8 |

**Operator decision rule applied (S5-reweighted):** no probe gates an
unconditional Phase 8 task. Tasks 0, 1 infra, 2, 3, 5, 7, and 8 ran
regardless. The three Class-B sub-tasks above each route to the
follow-up entries listed below.

---

## Deferred items

### 1. Cohort (a) — Phase 6 completion-gesture telemetry markers

- **Class:** Probe-miss no-op (Probe 1 + Probe 2).
- **Trigger:** No `'telemetry: deferred to Phase 8'` markers were
  seeded by upstream phases at the 2026-05-19 gate; no
  `verify_token` / `verify_share_token` symbols exist under
  `src/elspeth/web/`. Phase 6 (completion gestures) merged into
  RC5.2 (commit `dd20888f0`) but did not seed the
  `composer.share.token_verify_failure_total` or
  `composer.share.link_expiry_hit_total` emit sites that Phase 8
  Task 1 Sub-task 7d was scoped to harvest.
- **What's missing:** Two counter emits inside the future
  shareable-reviews verify-failure and expiry-hit branches:
  - `composer.share.token_verify_failure_total` (counter, no
    attributes per current B3 cohort-a design — review the
    attribute shape during Phase 9 design pass).
  - `composer.share.link_expiry_hit_total` (counter, no
    attributes).
- **Closure path:**
  1. Confirm with Phase 6's owner where the verify-failure and
     expiry-hit branches live (likely under
     `src/elspeth/web/shareable_reviews/`).
  2. File a Filigree issue under Phase 9 referencing this entry
     and the cohort-(a) design in
     `20-phase-8-polish-and-telemetry.md` §"Cross-phase telemetry
     — cohort split (B3 reshape)".
  3. Wire the emits with the existing 8b helper module shape
     (`src/elspeth/web/composer/telemetry_phase8.py` for module-local
     counters or `_SessionsTelemetry` container if cohort lives
     under sessions). Commit MUST include the
     `telemetry-backfill: phase-6` trailer (B4-r3 commit-msg
     hook enforces).
- **Definition of done:** The cohort-a counters increment under the
  same Q-cluster test discipline (function-scoped fixture per Q10)
  Phase 8 used for cohorts (b1) and (b2); plan §"Cohort attribution
  via commit trailers (A4 — load-bearing)" cited inline at the
  emit sites.

### 2. Task 6 — Tutorial-replay button (Phase 4 hello-world dependency)

- **Class:** Probe-miss no-op (Probe 6).
- **Trigger:** No `tutorial_completed` symbol exists under
  `src/elspeth/web/` at the 2026-05-19 gate. Phase 4 (hello-world
  tutorial) has not yet been planned (plan reference: see roadmap
  in `00-implementation-roadmap.md`).
- **What's missing:** The entire Task 6 surface from
  `20-phase-8-polish-and-telemetry.md`:
  - `src/elspeth/web/frontend/src/components/settings/TutorialReplayButton.tsx`
    and its `.test.tsx`.
  - Mount inside `ComposerPreferencesPanel.tsx`.
  - `updateComposerPreferences` call site in `api/client.ts`
    extended to clear the flag.
  - `preferencesStore.ts` action / selector for the flag.
  - The audit-side flag-clear path in the backend (depends on
    Phase 4's schema decision for where `tutorial_completed`
    actually lives).
- **Closure path:** Phase 4 ships the `tutorial_completed`
  surface; Phase 9 (or a Phase 4 follow-up) wires the replay button
  per the Phase 8 plan text. The plan text for Task 6 in
  `20-phase-8-polish-and-telemetry.md` remains canonical until then.
- **Definition of done:** A replay click PATCHes
  `{"tutorial_completed": false}` and the empty-state chat does
  NOT gain a "redo tutorial" link (explicit non-feature in Phase 8
  scope boundaries — Phase 9 should preserve that constraint).

### 3. Decision 2 — `composer.tutorial.replayed_total` counter boundary question

- **Class:** Decision deferral.
- **Trigger:** Pass-2 review surfaced that the
  `composer.tutorial.replayed_total` counter does not cleanly fit
  CLAUDE.md's "non-decision read" superset exception — the click
  is user-write-intent, not a read. Three options were considered:
  - Option A: emit the counter under the superset exception
    (rejected on pass-3 — would broaden the exception
    project-wide or require semantic dishonesty).
  - Option B: audit-record the replay (rejected by the original
    B2.b reasoning — account-level preferences are operational
    signal only).
  - Option C: defer the boundary question to Phase 9 (chosen).
- **What's missing:** A principled resolution. Either:
  - Phase 9 establishes a new project-wide rule that user-write-intent
    on operational-only surfaces can be telemetry-only (extending
    the CLAUDE.md superset exception with explicit scope), then
    Phase 8's plan text for Task 6 Step 7 is reactivated and the
    counter ships.
  - OR Phase 9 confirms the operational-only surface should
    promote to audit when user-write-intent appears, which
    triggers a Tier-1 schema decision (and a DB-delete) on the
    next deploy.
- **Closure path:** Phase 9 design pass; resolution amends both
  CLAUDE.md (if rule changes) and the Phase 8 plan text in place.
- **Definition of done:** The counter either ships under a named
  rule extension or is permanently retired. The plan name
  (`composer.tutorial.replayed_total`) stays in
  `20-phase-8-polish-and-telemetry.md` §"Telemetry primacy
  explicit acknowledgment" as a Phase-9 pointer until resolved —
  do not silently delete it; the deferral itself is the artifact.

### 4. `[all]` extra missing prometheus deps — resolver conflict with azure-monitor-opentelemetry-exporter

- **Class:** Decision deferral (dependency hygiene; pin-conflict resolution).
- **Trigger:** Phase 8a-3 (B1-r3 MeterProvider) added
  `opentelemetry-exporter-prometheus>=0.62b0,<1` and
  `prometheus_client>=0.21,<1` to the `webui` extra in
  `pyproject.toml`. They were intentionally **omitted from the
  `all` extra** because `azure-monitor-opentelemetry-exporter`
  pins `opentelemetry-sdk==1.40` exactly while
  `opentelemetry-exporter-prometheus` requires `>=1.41`. The
  packages are runtime-compatible (Prometheus exporter b62 works
  fine with OTel SDK 1.41.x) but the resolver rejects the
  declared metadata.
- **Current workaround:** Developers running `.[all]` must
  separately `uv pip install
  'opentelemetry-exporter-prometheus>=0.62b0,<1' 'prometheus_client>=0.21,<1'`.
  Documented in a comment block inside `pyproject.toml`'s `all`
  extra. The `webui` extra alone installs cleanly.
- **Closure paths (pick one in Phase 9):**
  1. Wait for `azure-monitor-opentelemetry-exporter` to relax its
     OTel SDK pin upstream. Track the upstream issue and re-add
     to `[all]` once it lands.
  2. Drop `azure-monitor-opentelemetry-exporter` from the `[all]`
     extra (move to a dedicated `azure-monitor` extra) so `[all]`
     can include Prometheus.
  3. File an upstream fix or pin override.
- **Definition of done:** `uv pip install -e ".[all]"` installs
  both the Azure exporter and the Prometheus exporter together,
  with no manual second `uv pip install` step.

### 5. Finding-7 self-healing-on-PATCH behavior reversed by B2 prior-load

- **Class:** Decision deferral / behavioral change disclosure.
- **Trigger:** Phase 8a-2 (B2 service-signature reshape, commit
  `417276bc2`) added a `_row_to_prefs` prior-load inside
  `preferences/service.py::update_composer_preferences` so the route
  has `prior.default_mode` in scope for the future Phase 8b
  account-level telemetry emit. As a side effect, a corrupt stored
  `default_composer_mode` row (CHECK constraint bypassed via PRAGMA
  manipulation or external mutation) that previously could be
  overwritten by a valid PATCH body now raises
  `CorruptPreferencesError` on the prior load, returning 500.
- **Why the change is the right answer per CLAUDE.md:** B1's audit
  payload now records `prior_trust_mode` honestly; from a corrupt
  prior row, no honest record is possible. Coercing or recovering
  would be silent fabrication (CLAUDE.md §"Three-Tier Trust Model"
  — Tier 1 data crashes on anomaly; the audit trail is the legal
  record). The operator's documented recovery path is delete-the-DB
  (`project_db_migration_policy.md` memory) or row-level repair.
- **What was deliberate before:** the pre-B2 architecture (test
  `test_patch_returns_written_values_not_a_reread`, write-path
  docstring) codified that a PATCH on a corrupt row should self-heal:
  the new write replaced the corrupt value without surfacing the
  prior corruption. That choice prioritised operator UX over Tier-1
  consistency.
- **Operator-visible implication:** rows that became corrupt before
  Phase 8 (e.g., from a PRAGMA-bypassed write in a prior session) now
  return 500 on the next PATCH instead of self-healing. Recovery is:
  identify the row, delete it (or delete the preferences DB), and
  PATCH again.
- **Phase 9 question:** is the Tier-1 alignment the right long-term
  answer, or should a one-shot "promote corrupt prior to NULL + fire
  audit event" recovery path exist? The corrupt-row scenario should
  be empirically rare (CHECK is enforced; the only known cause is
  PRAGMA manipulation), so the Tier-1 answer is probably correct.
  But the design question deserves an explicit gate rather than
  silent inheritance.
- **Closure path:** Phase 9 either (a) confirms the Tier-1 answer
  and removes this entry, or (b) designs and adds a corrupt-prior
  recovery seam (with audit event). Either way, the operator's
  decision should be recorded against this entry.

### 6. B3 cohort (c) — Performance instrumentation in shipped phases

- **Class:** Performance instrumentation.
- **Trigger:** Two perf signals were identified during the
  cross-phase telemetry split (B3 reshape) that belong to
  already-shipped phases:
  - `composer.audit.render_duration` (Phase 2C audit-readiness
    panel — measures the time from a row-detail fetch to first
    paint).
  - `composer.interpretation.resolve_duration` (Phase 5b
    interpretation resolve — measures latency from prompt commit
    to interpretation render).
- **Why deferred:** Pure perf signals; reopening shipped phases
  for perf instrumentation is out of proportion to the benefit
  and not security-relevant. Recorded here so the gap is
  discoverable rather than invented later.
- **Closure path:** Phase 9 perf-instrumentation pass adds these
  counters as histograms (NOT counters — duration is a distribution,
  not a count). Commits MUST include the appropriate cohort trailer
  (`telemetry-backfill: phase-2c` or `phase-5b`) for `git blame`
  discoverability.
- **Definition of done:** Histograms emit under the vocabulary
  Phase 8 established (sub-section names, naming conventions);
  dashboards visualise p50/p95.

### 7. Template card dynamic-source dispatch (Phase 5a wiring absent at Phase 8 gate)

- **Class:** Probe-miss no-op (Phase 5a dynamic-source-from-chat dispatch for
  template clicks not wired at the 2026-05-19 gate).
- **Trigger:** Phase 8c-3 replaced generic template cards with six
  audit-domain exemplars from README.md §"Example Use Cases". The new
  `TemplateCardsProps.onSelectTemplate` callback was given a second argument
  (`recommendedStartingPoint: ExampleUseCase["recommended_starting_point"]`)
  so the caller can route `"dynamic_source_from_chat"` cards through Phase 5a's
  inline-source creation path. At the Phase 8 gate, no dispatch function
  exists that accepts a seed prompt and fires the inline-source creation
  flow from a template click: Phase 5a's `InlineSourceCreatedTurn`,
  `InlineSourceDisambiguationTurn`, and `InlineSourceFallbackPrompt`
  components are shipped and wired for chat-turn rendering, but
  `ChatPanel.handleSelectTemplate` only feeds the seed prompt into the
  text input (`setInputText`). The `_recommendedStartingPoint` argument
  is accepted and discarded.
- **What's missing:** A dispatch branch in `ChatPanel.handleSelectTemplate`
  (or a new hook) that, when `recommendedStartingPoint === "dynamic_source_from_chat"`,
  calls the same path that wires a user-typed seed phrase into an
  inline-source creation request rather than populating the text input.
- **Affected files:**
  - `src/elspeth/web/frontend/src/components/chat/ChatPanel.tsx` —
    `handleSelectTemplate` (the `_recommendedStartingPoint` discard comment
    references this entry).
  - `src/elspeth/web/frontend/src/components/chat/TemplateCards.tsx` —
    the `onSelectTemplate` prop already carries the second argument; no
    change needed here once the caller is wired.
- **Closure path:**
  1. Confirm the correct API call / store action that triggers an
     inline-source creation from a chat message (the existing flow is
     triggered by a user sending a message, not from a template click).
  2. In `ChatPanel`, add a branch in `handleSelectTemplate`: if
     `recommendedStartingPoint === "dynamic_source_from_chat"`, call
     `sendMessage(seedPrompt)` (or the appropriate inline-source entry point)
     rather than `setInputText(seedPrompt)`.
  3. Remove the `_recommendedStartingPoint` discard comment and the
     leading underscore.
  4. Update `TemplateCards.test.tsx` and `ChatPanel.test.tsx` for the
     new dispatch behaviour.
- **Definition of done:** Clicking a `"dynamic_source_from_chat"` card fires
  the same inline-source creation path as typing and sending the seed
  prompt in the chat input. Cards with `"csv_upload"` or `"api_source"`
  continue to populate the text input.

### 8. Task 4 — "Show archived" toggle is dormant at runtime (backend soft-delete not implemented)

- **Class:** Decision deferral.
- **Trigger:** Phase 8c-4 added `archived?: boolean` to the `Session`
  frontend type and implemented a "Show archived" checkbox in
  `HeaderSessionSwitcher`. The toggle works correctly in tests (seeded
  via `setState`) but is dormant at runtime because:
  - `GET /api/sessions` does not return archived sessions. The backend
    `archive_session()` in `sessions/service.py` physically deletes the
    row; `list_sessions()` has no `include_archived` parameter.
  - `SessionRecord` has no `archived` field.
  - The frontend `Session` type's `archived?: boolean` field is never
    populated from real API responses.
- **What's needed for the toggle to be functional at runtime:**
  1. Change `archive_session()` in `sessions/service.py` to soft-delete
     (set an `archived_at` timestamp column instead of deleting the row).
     This is a schema change — requires a DB delete per
     `project_db_migration_policy.md`.
  2. Add `include_archived: bool = False` query parameter to
     `GET /api/sessions` route; filter on it in `list_sessions()`.
  3. Return `archived: bool` in `SessionResponse` (the pydantic model
     used by `_session_response()`).
  4. Update the frontend `Session` type: `archived` becomes
     non-optional (`archived: boolean`).
  5. Update `sessionStore.archiveSession`: the store action currently
     filters the archived session out of `state.sessions`; with
     soft-delete, it should keep the session but mark `archived: true`.
- **What is correct today:** The filter predicate, show-archived
  checkbox, Q9 error handling (await + catch + `role="alert"`), and
  the `archived?: boolean` type annotation are all correct and fully
  tested. No frontend rework is needed when the backend extends.
- **Closure path:** Phase 9 schema/backend pass implements soft-delete.
  The frontend changes in Step 4-5 above are the only frontend
  rework needed.
- **Definition of done:** Archived sessions appear in the switcher
  when "Show archived" is checked; active sessions are hidden when
  unchecked. The `GET /api/sessions?include_archived=true` probe
  returns at least one archived session in a fresh staging DB.

---

## A11y findings deferred from Phase 8 Task 7

> This section is **lazily created** by Task 7 Step 5/7 once axe-core
> has actually run. Phase 8 implementer: when Task 7 runs and produces
> medium/low-severity findings, append entries below using the
> template at the bottom of `20-phase-8-polish-and-telemetry.md`
> §"Accessibility (axe-core) findings deferred from Phase 8 Task 7".
> High-severity findings are fixed inline during Task 7 — do not log
> them here.

### Task 7 audit outcome (2026-05-19)

21 components audited with `jest-axe` + `axe-core@4.x` under WCAG 2.0/2.1
A/AA rule set (color-contrast disabled because jsdom does not compute
CSS-variable values — verify contrast against design tokens manually).

**Result: zero violations across all 21 components.** No high-severity
fixes were required, no medium/low findings remain deferred. The
component-level a11y discipline that Phase 1-7 carried (focus management,
role contracts, `aria-modal`, `useFocusTrap` reuse, role="alert"/"status"
regions, `aria-label` on icon-only buttons) is reflected in the audit
result.

Audited components:

- Phase 1B: ComposerPreferencesPanel, UserMenu, InlineOptOutCheckbox,
  DefaultModeChangedBanner
- Phase 2C: AuditReadinessPanel, ReadinessRowDetail, ExplainDialog
- Phase 3A/3B: AppHeader, HeaderSessionSwitcher, HeaderVersionSelector,
  SideRail, GraphMiniView, ExportYamlModal
- Phase 5a: InlineSourceCreatedTurn, InlineSourceDisambiguationTurn,
  InlineSourceFallbackPrompt
- Phase 6B: CompletionBar
- Phase 7B: PluginCard, FilterChipStrip
- Phase 8: TemplateCards (regrouped), ShortcutsHelp (four-group regroup)

Out of scope per Phase 8 review-findings: TutorialReplayButton (deferred
to Phase 9 in `19-phase-8-review.md`); the `expert` UI surfaces; the chat
message stream's full conversational state. These will be audited as part
of the Phase 9 closeout sweep when the deferred surfaces ship.

Manual follow-up required (NOT axe-detectable):

- **Color-contrast verification against design tokens.** jsdom cannot
  compute CSS-variable values, so the `color-contrast` rule was disabled.
  At the start of Phase 9, run an authentic-browser axe audit (e.g.
  `@axe-core/playwright` via a Playwright spec, or DevTools axe extension
  against staging) on the same component set to verify contrast.

_(no individual findings)_
