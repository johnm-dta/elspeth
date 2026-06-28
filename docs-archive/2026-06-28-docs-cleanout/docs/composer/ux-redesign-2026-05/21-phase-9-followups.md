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
| `grep -rn 'export_yaml\|Export YAML' src/elspeth/web/frontend/src/` | ≥8 | Task 1 Sub-task 7c (Phase 6 completion-gesture emit — composer.session.completed_total) | (HIT — scoped IN, wired post-review 2026-05-19) | n/a |
| `grep -rn 'auto_interpreted_opt_out' src/elspeth/web/` | ≥1 | Task 1 Sub-task 7e (B3 cohort b1 — Phase 5b interpretation-opt-out emit) | (HIT — scoped IN) | n/a |
| `grep -rn 'audit_readiness\|composer/audit/readiness' src/elspeth/web/` | ≥1 | Task 1 Sub-task 7f (B3 cohort b2 — Phase 2C audit-fetch-failure emit) | (HIT — scoped IN) | n/a |
| `ls src/elspeth/web/frontend/src/components/sessions/HeaderSessionSwitcher.tsx` | file exists | Task 4 (session-sidebar migration) | (HIT — scoped IN) | n/a |
| `grep -rn 'tutorial_completed' src/elspeth/web/ \| grep -v 'tutorial_completed_total' \| grep -v 'telemetry_phase8.py' \| grep -v 'sessions/telemetry.py'` | 0 | Task 6 (tutorial-replay button + counter) | Probe-miss no-op | Class B — scope OUT of Phase 8 |

**Operator decision rule applied (S5-reweighted):** no probe gates an
unconditional Phase 8 task. Tasks 0, 1 infra, 2, 3, 5, 7, and 8 ran
regardless. The three Class-B sub-tasks above each route to the
follow-up entries listed below.

**Post-review correction — Sub-task 7c (2026-05-19).** The Phase 8
overall-plan reviewer surfaced that Sub-task 7c (plan §2039–2047 —
the conditional Phase 6 YAML-export completion-gesture emit) was
silently missed: the `record_session_completed` helper landed at
`src/elspeth/web/composer/telemetry_phase8.py:248` and the
`session_completed_total` slot at
`src/elspeth/web/sessions/telemetry.py:198, 253, 317` were both
present, but the two call sites that actually write the
`composer_completion_events_table` audit row
(`shareable_reviews/service.py:319` for `mark_ready_for_review` and
`sessions/routes.py:5652` for `export_yaml`) never invoked the helper.
The probe shape above (`grep -rn 'export_yaml\|Export YAML'
src/elspeth/web/frontend/src/`) HITS — the frontend has long had the
Export YAML surface — so the plan's wire-it branch fires. The wire-up
was completed post-review per the plan §2039–2047 directive (no
deferral to Phase 9; the operator's "fix any issues you find
automatically" directive governs). The `_CompletionVerb` Literal was
also reconciled at the same time to mirror the DB CHECK constraint at
`src/elspeth/web/sessions/models.py:735` (eliminating the
`save_for_review` UI-vs-DB vocabulary drift and the `run_pipeline`
audit-row absence — see commit body for the superset-rule rationale).

---

## Deferred items

### 1. Cohort (a) — Phase 6 share-counter increment-site emits

> **Note (2026-05-19, post-review):** the heading was previously
> "Phase 6 completion-gesture telemetry markers", which conflated this
> cohort (the share-counter increment sites for
> `composer.share.token_verify_failure_total` and
> `composer.share.link_expiry_hit_total`, which depend on a Phase 6
> token-verify path that has NOT yet shipped) with Sub-task 7c (the
> completion-gesture counter `composer.session.completed_total`, whose
> emit sites at `mark_ready_for_review` and `export_yaml` HAD shipped
> by 2026-05-19 — the wire-up landed post-review). The body of this
> section is, and always was, about the share-counter cohort; the
> rename realigns the title with the body so a future reader does not
> assume the completion-gesture follow-up still applies. The
> Sub-task 7c follow-up is closed by the post-review wire-up commit;
> see the pre-flight gate probe table above for its row.


- **Class:** Probe-miss no-op (Probe 1 + Probe 2) — **counters defined,
  increment sites absent**.
- **Trigger:** No `'telemetry: deferred to Phase 8'` markers were
  seeded by upstream phases at the 2026-05-19 gate; no
  `verify_token` / `verify_share_token` symbols exist under
  `src/elspeth/web/`. Phase 6 (completion gestures) merged into
  RC5.2 (commit `dd20888f0`) but did not seed the increment-site
  call sites that Phase 8 Task 1 Sub-task 7d was scoped to
  harvest. **The counters themselves are not missing.** Counters
  `composer.share.token_verify_failure_total` and
  `composer.share.link_expiry_hit_total` are **defined and
  registered** at `src/elspeth/web/sessions/telemetry.py:327,331`
  inside `_SessionsTelemetry`. The follow-up is to add
  **increment-site call** in the verify-failure and expiry-hit
  branches under `shareable_reviews/` — the counters ship; what
  is missing is the `.add(1, attributes={...})` calls.
- **What's missing:** Two counter `.add(1, attributes={...})` call
  sites inside the future shareable-reviews verify-failure and
  expiry-hit branches. The counter objects already exist at
  `src/elspeth/web/sessions/telemetry.py:327` and `:331`; the
  follow-up is purely to wire the increments at the right code
  branches:
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
  3. Wire the **increment calls** at those branches (the counters
     are already registered on `_SessionsTelemetry` at
     `src/elspeth/web/sessions/telemetry.py:327,331`; emits go
     through `tel.share_token_verify_failure_total.add(1, ...)` and
     `tel.share_link_expiry_hit_total.add(1, ...)`). The 8b helper
     module pattern at `src/elspeth/web/composer/telemetry_phase8.py`
     is the canonical wrapper shape for the new emit functions.
     Commit MUST include the `telemetry-backfill: phase-6` trailer
     (B4-r3 commit-msg hook enforces).
- **Definition of done:** The cohort-a counters increment under the
  same Q-cluster test discipline (function-scoped fixture per Q10)
  Phase 8 used for cohorts (b1) and (b2); plan §"Cohort attribution
  via commit trailers (A4 — load-bearing)" cited inline at the
  emit sites.

### 2. Task 6 — Tutorial-replay button (Phase 4 hello-world dependency)

- **Class:** Probe-miss no-op (Probe 6).
- **Trigger:** No **persisted** `tutorial_completed` field exists
  under `src/elspeth/web/` at the 2026-05-19 gate. Phase 4
  (hello-world tutorial) has not yet been planned (plan reference:
  see roadmap in `00-implementation-roadmap.md`).
- **Probe shape (post-correction):** The probe targets persisted
  schema, not any string match. Two equivalent formulations:

  ```bash
  # Option A (exclusion — preferred):
  # Probe: no persisted tutorial_completed field in DB schema,
  # Pydantic models, or PATCH routes. The OTel counter
  # tutorial_completed_total at telemetry_phase8.py and
  # sessions/telemetry.py is a separate, intentional artifact —
  # exclude it from the probe.
  grep -rn 'tutorial_completed' src/elspeth/web/ \
    | grep -v 'tutorial_completed_total' \
    | grep -v 'telemetry_phase8.py' \
    | grep -v 'sessions/telemetry.py'
  # Expected: zero hits.
  ```

  ```bash
  # Option B (positive assertion — alternative):
  # Probe: no persisted tutorial_completed field exists.
  grep -E "tutorial_completed[^_]" src/elspeth/web/sessions/models.py \
    src/elspeth/web/preferences/models.py \
    src/elspeth/web/preferences/routes.py
  # Expected: zero hits. The boundary character class `[^_]`
  # excludes tutorial_completed_total and tutorial_completed_at
  # (the timestamp column, which IS persisted but is the correct
  # shape).
  ```

  The OTel counter `tutorial_completed_total` (registered at
  `composer/telemetry_phase8.py:78` and emitted from
  `sessions/telemetry.py:191`) IS a legitimate telemetry artifact
  for Phase 8 — the corrected probe above explicitly excludes it.
  The Phase 9 follow-up is the **persisted-field** absence, not the
  telemetry-counter absence.
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
- **Note (2026-05-19, CR-1):** This decision concerns
  `composer.tutorial.replayed_total` specifically — the
  replay-button counter. It is **NOT** affected by Phase 4's
  `composer.tutorial.completed_total` emit; that counter slot was
  already declared in Phase 8 (`sessions/telemetry.py:317-323`)
  and Phase 4 (`21a2-phase-4-backend-part-2.md` Task 8, when Phase 4
  ships) provides the emit site. Phase 4 attaches a
  `completion_path` attribute (values
  `first_time | skip | retake | repeat`) to discriminate the four
  PATCH gestures server-side. Phase 4's emit does NOT resolve the
  `replayed_total` boundary question below — `replayed_total`
  remains Phase-9-deferred per Option C.
- **Status in code (load-bearing absence markers):** The counter
  `composer.tutorial.replayed_total` has no runtime emit site,
  which is correct per the Phase 9 deferral. The string appears in
  two **deliberate-absence markers** at
  `composer/telemetry_phase8.py:234` and `sessions/telemetry.py:192`
  (both labelled `# DELIBERATELY ABSENT — Phase 9 deferred…`);
  these are load-bearing artifacts
  that ensure a Phase 9 reader sees the deferral intent and does
  not silently add the emit prematurely. They are not regressions
  and must not be removed when this entry is closed — they remain
  until the Phase 9 resolution lands a real emit (or formally
  retires the counter).
- **Trigger:** Pass-2 review surfaced that the
  `composer.tutorial.replayed_total` counter does not cleanly fit
  the `logging-telemetry-policy` skill's Superset Rule under
  account-level preferences — the click is user-write-intent, not
  a read. Three options were considered:
  - Option A: emit the counter as an operational-metrics exemption
    (rejected on pass-3 — would broaden the exemption
    project-wide or require semantic dishonesty).
  - Option B: audit-record the replay (rejected by the original
    B2.b reasoning — account-level preferences are operational
    signal only).
  - Option C: defer the boundary question to Phase 9 (chosen).
- **What's missing:** A principled resolution under the
  `logging-telemetry-policy` skill's framing. Either:
  - Phase 9 establishes a new project-wide rule that user-write-intent
    on operational-only surfaces can be telemetry-only under the
    **operational metrics exemption** (`lacks probative value` for
    an individual decision; only the aggregate matters). Then
    Phase 8's plan text for Task 6 Step 7 is reactivated and the
    counter ships. Per the `logging-telemetry-policy` skill, the
    **Superset Rule** (every telemetry event must be in the audit
    superset) is satisfied because tutorial replay is already
    audited via the Landscape entry's `seeded_from_cache: true`
    marker (Phase 4 P2); the counter itself is deferred under the
    operational metrics exemption — only the aggregate matters,
    which Phase 9 will surface as a hit-rate signal.
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
  against staging) on these four surfaces and record any violations as
  Phase 9 follow-up entries in this file (do not block Phase 8 close on
  contrast findings — per plan §20 lines 3594–3609, this is the
  compensating manual control for Task 7's jsdom-blind axe suite):
    - **Empty-state / template cards (Task 3).** The new audit-domain
      template cards rendered when no session is active; verify card
      backgrounds and SDA-list text contrast.
    - **Audit-readiness panel (Phase 2C surface).** Text colour against
      the readiness-row backgrounds; severity indicator colour against
      row backgrounds.
    - **Header session switcher (Task 4).** Filter input placeholder
      text; archive-button hover and focus states.
    - **Settings panel — `TutorialReplayButton` (Task 6 surface, audit
      deferred to Phase 9 per `19-phase-8-review.md`).** Enabled,
      disabled, and `role="status"` confirmation states.

_(no individual findings)_

---

## §8 — Frontend lint debt (pre-existing pre-Phase-8, surfaced by Phase 8.8 sweep)

**Class:** Probe-miss no-op (pre-existing — none of these files were
touched by Phase 8). Surfaced by `npm run lint` during the Task 8 final
sweep on 2026-05-19.

**Closure path:** dedicated frontend-lint sweep in Phase 9 (Class B — out
of scope for Phase 8 polish per `feedback_no_scope_dumping.md` only if
they are pre-existing and unrelated to Phase 8 work; verified by
`git log RC5.2..HEAD -- <file>` returning empty for each).

### F8.1 — `@typescript-eslint/no-explicit-any` rule missing (6 errors)

Files (none touched by Phase 8 per `git log RC5.2..HEAD -- <file>`):

- `src/components/execution/InlineRunResults.test.tsx:31,33,35,208,249`
- `src/hooks/useNarrativeMode.test.ts:1`

Root cause: ESLint config uses `@typescript-eslint/no-explicit-any` rule
without loading the `@typescript-eslint` plugin in the `eslint.config.js`
flat-config. The rule definitions exist but the plugin is not registered
under that namespace — so files that `/* eslint-disable
@typescript-eslint/no-explicit-any */` (the way both files do) report
"Definition for rule not found."

Fix shape: either (a) wire `@typescript-eslint` plugin into
`eslint.config.js` so the rule resolves and the disable comments work
silently, or (b) replace the comment with a generic ESLint comment that
the current config recognises, or (c) refactor the tests to avoid `any`.

### F8.2 — `react-hooks/exhaustive-deps` warnings (4 warnings)

Files (none touched by Phase 8 per `git log RC5.2..HEAD -- <file>`):

- `src/components/catalog/CatalogDrawer.tsx:266` — useCallback
  unnecessary dependency `schemaErrors`
- `src/components/chat/ChatInput.tsx:127` — useEffect missing
  dependency `inputRef`
- `src/components/inspector/GraphView.tsx:150,155` — useCallback
  unnecessary dependency `resolvedTheme` (two sites)

Each requires per-site analysis — exhaustive-deps warnings are not
mechanical (adding the suggested dep can introduce re-render loops; the
"correct" answer is usually one of: refactor the closure, add the dep
and verify, or assert with a justification comment).

### Phase 9 closeout note

Phase 8.8's brief was "Phase 8 must not check in any lint warnings." The
six errors and four warnings above are **not Phase-8-introduced** —
`git log RC5.2..HEAD -- <file>` returns empty for every file. The
implementer therefore preserved them as pre-existing debt rather than
expanding Phase 8 scope into a frontend-lint sweep. Phase 9 should
address them as a dedicated sweep.

---

## §F8.3 — /metrics route precedence regression (FIXED inline in Phase 8.8)

Recorded for traceability — not a Phase 9 follow-up.

Phase 8a-3 (`e963e725f`) wired `app.mount("/metrics", make_asgi_app())`
in `src/elspeth/web/app.py`. Starlette `Mount` instances only match the
exact prefix with a trailing slash; bare `/metrics` falls through. In
production, the SPA `StaticFiles(html=True)` mount at `/` then catches
the request, finds no `metrics` file in `dist/`, and returns 404. The
test `tests/unit/web/test_meter_provider.py::
test_metrics_endpoint_returns_prometheus_format` (added in the same
commit) only surfaces this when `dist/` exists; Phase 8a-3 shipped
green because the backend test suite runs without a frontend build.

Phase 8.8 final sweep ran `npm run build` (Step 4) which created
`dist/`, then ran the backend tests (Step 5), exposing the regression.
Fix landed in this commit: replace `app.mount("/metrics", ...)` with a
`@app.get("/metrics")` route handler that calls `generate_latest()`
from `prometheus_client` directly. Route handlers match before mounts
in Starlette regardless of registration order, so this resolves
cleanly against the SPA catch-all.

Verification: `test_metrics_endpoint_returns_prometheus_format` 404 →
200; full unit-test suite 16504 → 16506 passed; ruff + mypy clean.

---

## §9 — Phase 9 follow-ups file line count exceeds soft cap (S4)

The plan's S4 soft-cap on this file is 30 lines. At Phase 8 close it
exceeds 380. This is acknowledged: the file documents real deferred
work from a 9-week effort and conciseness was deprioritised in favour
of evidence-bearing entries that close cleanly. The cap exists so the
file does not become an undifferentiated debt-dump; the entries here
each carry a closure path, class, and audit trail. Phase 9 entry-point:
treat the §1–§9 headers as the menu, promote the highest-severity items
to Filigree issues, and either retire or carry forward the rest at the
90-day TTL.
