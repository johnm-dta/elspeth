# Phase 5b — Surface the LLM's interpretation (overview)

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development`
> (recommended) or `superpowers:executing-plans` to implement this plan task-by-task.
> Phase 5b is split across two sibling plans because it touches the audit
> schema, a new recorder method, a new composer tool, two frontend turn
> widgets, and audit-panel integration — the file would exceed the 1500-line
> ceiling as a single document.

This file is the **overview + B2 verdict + cross-cutting policy**. The
task-by-task implementation is in:

- [18a-phase-5b-backend.md](18a-phase-5b-backend.md) — schema migration,
  recorder method, composer tool, service wiring, runtime prompt-template
  hand-off, redaction.
- [18b-phase-5b-frontend.md](18b-phase-5b-frontend.md) — turn widget,
  freeform inline message, audit-readiness panel row, integration test,
  composer-skill nudge, staging smoke.

The two plans MUST be implemented in order (18a then 18b): the frontend
consumes the new tool surface, the new event payload shape, and the new
audit-panel API field introduced in 18a.

---

## B2 verdict — (c) New event class → new schema table + recorder method + composer-service wiring

Open question B2 (per
[00-implementation-roadmap.md](00-implementation-roadmap.md) §"Pre-Phase-5b
blocks: surface-the-LLM's-interpretation") asked: **does the audit recorder
support interpretation-acceptance events?**

The reconnaissance below classifies the answer as **(c) new event class**.
The current audit infrastructure has no analog that fits the six required
fields cleanly; extending an existing event type would force the
interpretation event into a shape that doesn't carry its semantics. A new
schema table is the honest representation.

### Recon notes (citations)

1. **Runtime Landscape (`core/landscape/data_flow_repository.py`) is the
   wrong audit DB for this event.** The Landscape records
   *pipeline-execution* events — rows, tokens, calls, validation errors,
   transform errors. Its `create_*` and `record_*` methods are at
   `src/elspeth/core/landscape/data_flow_repository.py` lines
   `381` (`create_row`), `467` (`create_token`), `874`
   (`record_token_outcome`), `1462` (`record_validation_error`), `1587`
   (`record_transform_error`). None of them model "a composition-time
   decision the user made before the run started." The Landscape has no
   row representing the *composer*; it has rows representing the
   *pipeline*. The interpretation-acceptance event happens at composition
   time and never reaches the runtime Landscape directly; what flows into
   the runtime is the *resolved prompt-template string* embedded in the
   committed `composition_state.nodes` JSON. The runtime audit chain then
   records that string as part of the LLM-transform's plugin config in
   the normal `calls` table, with the standard `arguments_hash` /
   `result_hash` envelope. The B2 question therefore lives entirely in
   the **session audit DB** (`web/sessions/`), not the Landscape.

2. **Closest composer-side analog is `proposal_events_table`.** Defined
   at `src/elspeth/web/sessions/models.py` line `342-372`. The table is
   a generic event log scoped to a session, with a closed `event_type`
   enum:

   ```text
   event_type IN ('proposal.created', 'proposal.accepted',
                  'proposal.rejected', 'trust_mode.changed')
   ```

   The CHECK constraint is closed by design — see the explicit "NO
   SILENT EXTENSION" block at `models.py` lines `274-289` (on
   `composition_states.provenance`, the parallel closed-enum table).
   Phase 5b cannot append `'interpretation.accepted'` here without an
   amendment to the closed enum, a writer-path audit, and a separate
   payload schema. That extension would also conflate two distinct event
   classes: `proposal_events` describes the lifecycle of a
   `composition_proposals` row (a staged tool call), whereas an
   interpretation event has no staged tool call and no proposal row —
   it is its own decision class.

3. **`composition_proposals_table` does not fit the shape either.**
   Defined at `models.py` line `291-340`. The required fields are
   `tool_call_id`, `tool_name`, `arguments_json`,
   `arguments_redacted_json`, `affects` — all of which are oriented
   around the LLM's tool-call lifecycle. The interpretation event has
   none of these: there is no tool call being staged; the LLM is
   *surfacing its own current best-guess interpretation of a subjective
   term* and asking the user to accept or amend it. Reusing the
   proposals table would force us to invent ghost `tool_call_id` values,
   would lose the `user_term` / `llm_draft` / `accepted_value` triple
   under generic `arguments_json`, and would make the audit-side query
   pattern ("show me all interpretation decisions for this session")
   require fragile event_type filtering instead of being a table-scoped
   query.

4. **`web/composer/audit.py` audits *tool dispatch*, not user decisions.**
   `BufferingRecorder` (`audit.py` line `185`) buffers
   `ComposerToolInvocation` records that land as `role=tool`
   `chat_messages` rows during the compose loop (see the module
   docstring at `audit.py` lines `1-39`). This is the wrong recorder
   class for the interpretation event because (a) the interpretation
   event is *not* a tool invocation — it is a user choice in response to
   a tool invocation; (b) the per-invocation envelope captures
   `arguments_canonical`, `result_canonical`, and a status discriminant
   (`SUCCESS / ARG_ERROR / PLUGIN_CRASH`) that have no natural mapping
   to the interpretation-event semantics; (c) the user's acceptance is
   not an LLM-side action and must be recorded with the user as actor,
   from a different code path (the route handler that receives the
   user's "Use mine / Change it" click) — not the compose loop.

5. **The six required fields from design-spec §"Recording the
   interpretation"** are: (a) user-provided term ("cool"), (b) LLM's
   draft interpretation, (c) accepted or amended interpretation, (d)
   timestamp, (e) user identity, (f) pipeline-state reference. The
   new event class also records: (g) `model_identifier`, (h)
   `model_version`, (i) `provider` (binding the row to the specific
   LLM that produced the draft), (j) `composer_skill_hash` (SHA-256
   of the skill markdown loaded into the composer LLM's context at
   call time), (k) `arguments_hash` (canonical rfc8785 hash over the
   row's required fields), and (l) `interpretation_source` (structural
   discriminant — `user_approved` | `auto_interpreted_opt_out` |
   `auto_interpreted_no_surfaces` — written by the service path, not
   the LLM). None of the original six, nor the new five, are
   first-class columns on any existing table. Inventing them as JSON
   sub-fields of an existing event payload sacrifices query ergonomics,
   audit-tooling readability, and the spec's stated intent that
   interpretation events be "discrete events."

### Verdict (c) implications

Phase 5b's backend scope is:

- A **new `interpretation_events_table`** in `web/sessions/models.py`,
  with first-class columns for the six required fields plus: a closed
  enum on the choice discriminant (`accepted_as_drafted` /
  `amended` / `opted_out`); LLM-provenance columns (`model_identifier`,
  `model_version`, `provider`, `composer_skill_hash`, `arguments_hash`)
  that bind the row to the specific LLM call that produced the draft and
  make the row replayable across model upgrades; and a structural
  `interpretation_source` discriminant (closed enum: `user_approved` |
  `auto_interpreted_opt_out` | `auto_interpreted_no_surfaces`) written
  by the service path — not the LLM — so the audit trail can
  definitively distinguish a user-approved interpretation from an
  auto-interpreted one without relying on a comment the LLM may or may
  not have emitted.
- A **new recorder method** `record_interpretation_event` on the
  session service, called from a **new composer tool** (per design-spec
  §"Implementation notes": "needs a tool that records an
  LLM-interpretation acceptance event as a distinct audit record,
  separate from the prompt template's plugin config").
- A **new LLM-callable tool** `request_interpretation_review` that the
  composer model invokes when its surfacing heuristic fires; the tool's
  result is a `ToolResult` that carries the in-flight interpretation
  state forward, with the user's response handled by a separate
  user-driven route (analogous to the `composition_proposals` /
  `proposal_events` two-phase lifecycle).
- A **runtime hand-off**: the user's accepted interpretation is bound
  into the affected LLM transform's prompt-template plugin config when
  the proposal commits, so subsequent runs use the resolved string
  deterministically and the runtime Landscape audit chain hashes it as
  part of the normal `calls` envelope. The runtime is never asked to
  re-interpret.

The frontend scope is described in detail in
[18b-phase-5b-frontend.md](18b-phase-5b-frontend.md).

---

## Goal

Land the user-visible affordance described in design doc
[06 §Feature 2](06-chat-as-data-entry.md#feature-2--surface-the-llms-interpretation).
After Phase 5b ships:

- When the composer LLM operationalizes a **subjective or underspecified
  user term** (e.g., "rate how cool they are"), it pauses the
  composition, surfaces its draft interpretation in a reviewable
  affordance, and waits for the user to accept or amend it before the
  pipeline is finalised.
- The user's accepted or amended interpretation is recorded in the
  session audit DB as a discrete event with all required fields,
  including LLM-provenance columns and a structural
  `interpretation_source` discriminant.
- The accepted interpretation flows into the affected LLM transform's
  prompt template **at composition time** so runtime behaviour is
  deterministic against the recorded value. Subsequent runs of the
  same pipeline do NOT re-interpret.
- The audit-readiness panel surfaces a new conditional row ("LLM
  interpretations: pending / accepted / none") that lets the operator
  see at a glance whether all surfaced interpretations have been
  resolved.

## Architecture

The interpretation-surfacing flow is a **two-phase, user-mediated
decision lifecycle**, structurally parallel to the existing
`composition_proposals` flow but with semantically distinct payload
shape:

```text
  ┌─── compose loop ───────────────────────────────────────────────┐
  │  LLM detects subjective term in user request                    │
  │       ↓                                                          │
  │  LLM calls request_interpretation_review(user_term, draft,      │
  │                                          pipeline_state_id,     │
  │                                          affected_node_id)      │
  │       ↓                                                          │
  │  Tool handler:                                                   │
  │    - validates arguments (Tier-3 boundary)                      │
  │    - persists pending interpretation event                      │
  │    - returns ToolResult with status=AWAITING_USER               │
  │       ↓                                                          │
  │  Compose loop ends; assistant message references the pending    │
  │  interpretation. Frontend renders the "Use mine / Change it"    │
  │  affordance.                                                     │
  └──────────────────────────────────────────────────────────────────┘

  ┌─── user-driven route ──────────────────────────────────────────┐
  │  User clicks "Use mine" or "Change it: <text>"                  │
  │       ↓                                                          │
  │  POST /api/sessions/{id}/interpretations/{event_id}/resolve     │
  │       ↓                                                          │
  │  Route handler:                                                  │
  │    - validates user choice (Tier-3 boundary)                    │
  │    - commits accepted_value into interpretation_events row     │
  │    - patches the affected LLM transform's prompt-template      │
  │      plugin config in composition_states.nodes (NEW state ver) │
  │    - records the state advance with provenance="interpretation │
  │      _resolve" (new closed-enum value; see §"Schema impact")    │
  │    - returns the new composition state                          │
  │       ↓                                                          │
  │  Frontend re-renders graph + audit panel; banner clears.        │
  └──────────────────────────────────────────────────────────────────┘

  ┌─── pipeline execution (later) ─────────────────────────────────┐
  │  Pipeline runs against the committed composition state.        │
  │  LLM transform's prompt is the resolved string; runtime never  │
  │  re-interprets. Landscape audit chain hashes the resolved     │
  │  prompt as part of the normal `calls` table envelope.          │
  └──────────────────────────────────────────────────────────────────┘
```

The flow shares no code with the runtime Landscape directly. The
runtime inherits the user's decision via the bound prompt-template
string — the runtime Landscape audits the run, the session audit DB
audits how the pipeline arrived at its current shape, and the
attributability test holds: `explain(recorder, run_id, token_id)` for
any output traces back through the runtime `calls` row → the
prompt-template string → the `interpretation_events` row recording
how that string came to be → the user who accepted it.

## Tech Stack

- Python 3.13, SQLAlchemy Core (session audit DB), pydantic (wire
  schemas + tool argument validation).
- React + Zustand + Vitest + testing-library for the frontend.
- The composer skill (Markdown prompt) for the LLM-side heuristic
  nudge.
- No new runtime dependencies; no new MCP-server surface; no Landscape
  schema changes.

## Sibling plans

| Plan | Status | Relationship |
|------|--------|---|
| [17-phase-5a-dynamic-source-from-chat.md](17-phase-5a-dynamic-source-from-chat.md) | Implementation in progress / ready to ship | **Prerequisite**. Phase 5a establishes the canonical hero example (`"create a list of 5 government web pages and use an LLM to rate how cool they are"`) end-to-end through inline-source-from-chat. Phase 5b layers the *second* feature onto the same hero example — the "cool" interpretation. The plans must integrate cleanly: Phase 5b's frontend turn widget must coexist in the same `ChatPanel.tsx` dispatch as Phase 5a's `InlineSourceCreatedTurn`. |
| [04-first-run-tutorial.md](04-first-run-tutorial.md) (Phase 4) | Dependent | Phase 4's hello-world tutorial *consumes* Phase 5b. The tutorial's third beat is the interpretation review of "cool"; until Phase 5b ships, the tutorial cannot teach that affordance. Phase 5b must ship before Phase 4 is plannable. |
| [14-phase-2-audit-readiness-panel.md](14-phase-2-audit-readiness-panel.md) | Already shipped (or in flight) | Phase 5b adds a **new conditional row** to the audit-readiness panel: "LLM interpretations: pending / accepted / none". If Phase 2 has shipped at planning time, Phase 5b's Task 18b-7 wires the row in. If not, the row is deferred to a Phase-2 followup and noted on the umbrella PR. |
| [16-phase-7-catalog-reshape.md](16-phase-7-catalog-reshape.md) | Independent | No interaction. |

## Scope boundaries

**In scope (Phase 5b):**

- New `interpretation_events_table` in the session audit DB with the
  six required fields; LLM-provenance columns (`model_identifier`,
  `model_version`, `provider`, `composer_skill_hash`, `arguments_hash`);
  a structural `interpretation_source` discriminant (closed enum:
  `user_approved` | `auto_interpreted_opt_out` |
  `auto_interpreted_no_surfaces`); and a closed `choice` enum
  (`accepted_as_drafted`, `amended`, `opted_out`).
- New session service methods: `create_pending_interpretation_event`,
  `resolve_interpretation_event`, `list_interpretation_events`,
  `record_session_interpretation_opt_out`.
- New composer tool `request_interpretation_review` (LLM-callable) that
  stages the pending event.
- New HTTP route `POST /api/sessions/{id}/interpretations/{event_id}/resolve`
  that commits the user's choice and patches the affected LLM
  transform's prompt template.
- New HTTP route `POST /api/sessions/{id}/interpretations/opt_out` that
  records the per-session "stop asking" decision as a row in
  `interpretation_events_table` with `choice='opted_out'` and
  `interpretation_source='auto_interpreted_opt_out'`. The session flag
  `interpretation_review_disabled = true` is also set on the session
  row as a fast-path read guard, but the `interpretation_events` row
  is the canonical audit record. The session flag must never be set
  without a corresponding audit row.
- New `composition_states.provenance` enum value `interpretation_resolve`
  (closed-enum extension — requires the documented "NO SILENT EXTENSION"
  governance step per `models.py` lines `274-289`).
- New frontend turn widget `InterpretationReviewTurn.tsx` for guided
  mode and a new inline-message variant in freeform mode (per design-spec
  §Risks: "the surfacing UI may differ").
- New audit-readiness panel row: "LLM interpretations: pending / accepted
  / none" (conditional on Phase 2 having shipped; see Task 18b-7).
- A composer-skill prompt nudge teaching the LLM when to call
  `request_interpretation_review` (subjective adjective heuristics from
  design-spec §"When the interpretation gets surfaced").
- A Vitest integration test asserting the end-to-end flow: LLM calls the
  tool → pending row created → user clicks "Use mine" → state version
  advances → audit row records all required fields.
- A Landscape spot-check Python test (verdict (c) **MANDATES** this per
  CLAUDE.md attributability test): query the session audit DB and assert
  the recorded event row contains all required fields (including the
  LLM-provenance columns and `interpretation_source`) with the correct
  types and the pipeline-state reference resolves to a real
  `composition_states` row.

**Out of scope (deferred to a later phase):**

- **The "edit a recorded interpretation later" workflow** (design-spec
  §"Editing the interpretation later"). This is a separate UX flow that
  requires graph-node selection plumbing and a "history of interpretations"
  view. Track as a Phase 5b-followup observation; do NOT bundle.
- **Heuristic tuning telemetry.** Design-spec §"When the interpretation
  gets surfaced" recommends biasing toward false positives and tuning
  with usage data. The telemetry signal that says "user opted out / user
  accepted-as-drafted / user amended" is *available* from the audit DB
  by simple SQL — exposing it on a dashboard is a Phase 11 observability
  concern, not part of Phase 5b's MVP.
- **Multi-turn interpretation editing.** If the user clicks "Change it",
  amends, then realises their amendment is wrong and wants to amend again
  within the same surfacing event, the MVP treats the first
  user-submitted value as final (no rolling-edit). A second surfacing
  for the same term within the same composition produces a new
  interpretation event (the design spec §"Editing the interpretation
  later" already says "a new audit record; the old interpretation is
  preserved in history"). Track as a deferred-polish observation.
- **The "stop asking" toggle's per-pipeline persistence.** The opt-out
  is session-scoped. Persisting it across sessions (e.g., as a user
  preference) is a Phase 8 polish item.
- **A composer-skill-level rate limiter for interpretation surfacing.**
  The heuristic in the skill prompt is intentionally biased toward
  false positives; tightening with rate-limit logic is a Phase 8 task
  once usage data exists.
- **Runtime Landscape extension for "interpretation provenance."** The
  runtime Landscape already audits the resolved prompt-template string
  via the normal `calls` envelope. Adding a backlink from the runtime
  `calls` row to the session `interpretation_events` row would let an
  auditor walking from a runtime call back to the composition-time
  decision skip a join, but the join through `composition_states.nodes`
  is already authoritative. Mark as a Phase 11 audit-tooling
  enhancement; do NOT bundle into Phase 5b.
- **Per-row redaction of `user_term` / `llm_draft` / `accepted_value`
  content.** Retention follows session-level deletion until then; Phase
  11 audit-tooling will introduce a redaction event class parallel to
  the audit-pack discipline.
- **Hash-chaining of `interpretation_events` rows for tamper
  detection.** The new event class is the first session-DB table whose
  contents materially affect runtime behaviour (the resolved prompt
  template). Tracked as a Phase 11 audit-tooling enhancement.

## Trust-tier check (mandatory before any data-handling work)

Phase 5b touches **three distinct trust boundaries** and the discipline
differs at each. This section is the canonical reference; the per-task
sections in 18a and 18b refer back to it.

| Boundary | Direction | Trust tier | Discipline |
|---|---|---|---|
| LLM → composer service | Tool-call arguments crossing into `request_interpretation_review` | **Tier 3 (external)** | Pydantic-validated at the tool boundary. Type-and-range coerce where safe (`user_term` is a `str` with a length cap; `llm_draft` is a `str` with a length cap; `affected_node_id` is validated to exist in the current composition state). Reject (ARG_ERROR) anything that doesn't conform. **Never persist the LLM's draft without surfacing it for user review** — that's the entire feature. |
| User → backend route | User's "Use mine / Change it" body crossing into `/resolve` | **Tier 3 (external)** | Pydantic-validated at the route boundary. The user's amended text is a `str` with a length cap (8 KiB after stripping) and strict content validation: reject `{{` or `}}` (template-injection guard), control characters (U+0000–U+001F except tab), and multi-line input. A placeholder-position check further requires that any `{{interpretation:…}}` token in the downstream prompt template sits in a value-position, not inside a `system:` / `role:` / `instructions:` section header. The choice discriminant is a closed enum (`accepted_as_drafted` / `amended` / `opted_out`); reject anything else. |
| Session audit DB → composer + frontend | Reading back a persisted interpretation event | **Tier 1 (ours)** | Read into a frozen dataclass (`InterpretationEventRecord`); crash on any anomaly (NULL in a NOT-NULL column, invalid enum value, dangling FK). No coercion. No `.get()`-with-default. The dataclass carries all columns: the six required fields, `model_identifier` / `model_version` / `provider` / `composer_skill_hash` / `arguments_hash`, and `interpretation_source`. |

**Key tier-discipline implications:**

1. **The LLM's `llm_draft` is Tier-3 until the user accepts it.** Once
   the user accepts (or amends), the `accepted_value` column becomes
   Tier-1 audit data. The DB row stores BOTH `llm_draft` (the original
   Tier-3 draft, recorded for forensic value) AND `accepted_value` (the
   Tier-1 user-approved string). The runtime prompt-template uses
   `accepted_value`, never `llm_draft`.

2. **The user's `accepted_value` is subject to strict content
   validation at the route boundary.** The composer route rejects
   `{{` / `}}` (which would allow a user to inject or clobber template
   placeholders), control characters (U+0000–U+001F except tab), and
   multi-line input. Length is capped at 8 KiB after stripping — the
   same upper bound as `chat_messages.content`. A placeholder-position
   check ensures the `{{interpretation:…}}` token in the downstream
   prompt template sits in a value-position, not inside a `system:` /
   `role:` / `instructions:` section header — preventing a user-supplied
   string from silently overriding a system prompt. These constraints are
   enforced before the value is persisted; the runtime tier-discipline
   wraps the interpolation as defence-in-depth, not as the primary
   guard.

3. **Reading the interpretation event back is Tier-1.** The
   `InterpretationEventRecord` dataclass has direct field access (no
   `.get()`), uses `freeze_fields()` per the project's
   `deep_freeze` contract, and crashes loudly on any read anomaly.

4. **The runtime Landscape's hash of the resolved prompt-template string
   is the integrity anchor.** If an attacker were to tamper with the
   `interpretation_events.accepted_value` column post-resolution, the
   tampering would NOT be detected by the runtime Landscape (which
   hashes only the resolved prompt string, not the upstream provenance).
   The session audit DB must therefore treat the `accepted_value`
   column as append-only after resolution; updates after `resolved_at`
   is set are rejected at the writer boundary. The `chat_messages`
   table has no UPDATE permission either; the same posture applies
   here.

5. **The `interpretation_source` discriminant is written by the service
   path, not the LLM.** An auditor asking "did the user approve this
   interpretation?" reads a structurally-enforced enum value, not a
   comment the LLM may or may not have emitted. The three values are:
   `user_approved` (the user saw the draft and accepted or amended it),
   `auto_interpreted_opt_out` (the user had previously opted out of
   review for the session), and `auto_interpreted_no_surfaces` (the
   LLM produced the interpretation inline without surfacing a review
   widget, e.g. because no subjective term was detected). This column
   is mandatory and NOT NULL; the writer path sets it before persisting
   the row.

**Telemetry posture.** Per CLAUDE.md's primacy contract, every audit-write path declares its telemetry posture explicitly. Interpretation event writes are audit-primary (Tier 1, user-decision-class) and emit NO operational telemetry — they are not ephemeral operational signals, they are durable legal records of user intent. The absence of telemetry at these call sites is a design decision, not an omission. This posture is recorded per-method in the 18a Task 4–7 service docstrings.

## Verification approach

Each backend task in 18a and each frontend task in 18b is TDD-shaped:
write the failing test → run it red → implement → run it green →
commit. The Landscape (session-audit-DB) spot-check is **required**
for every backend task that records to the new table (per CLAUDE.md
attributability test). The spot-check shape is a small fixture that:

1. Drives the writer path (calling the service method or POSTing to the
   route).
2. Opens a read-only SQLAlchemy connection to the session audit DB.
3. Issues a `SELECT * FROM interpretation_events WHERE id = :id` and
   asserts each of the six required fields, the LLM-provenance columns
   (`model_identifier`, `model_version`, `provider`,
   `composer_skill_hash`, `arguments_hash`), and `interpretation_source`
   is present, of the expected type, and matches the expected value.
4. Verifies the FK to `composition_states` resolves (i.e., the
   pipeline-state reference is real, not dangling).

Empirical validation of the surfacing heuristic (whether the LLM
correctly calls `request_interpretation_review` for "rate how cool they
are" but not for "rate as numeric 1-10") is **not** test-gated against
skill-prompt text. The composer-skill markdown is an LLM prompt, not
code (per the project memory entry `feedback_no_tests_for_skill_prompts`).
Validation is empirical: re-run the canonical hero prompt through the
live LLM on staging after Phase 5b ships and verify the LLM surfaces
"cool" but not "5" or "1-10".

**Manual smoke at the end of the plan (Task 18b-9):**

1. Start a fresh session on staging (`elspeth.foundryside.dev` per the
   project memory `project_staging_deployment`). Restart
   `elspeth-web.service` after any skill-prompt edit.
2. Type the canonical hero prompt: "create a list of 5 government web
   pages and use an LLM to rate how cool they are".
3. Confirm Phase 5a's inline-source path fires first (5 URLs surfaced
   for review).
4. Confirm Phase 5b's `InterpretationReviewTurn` then fires for "cool",
   with the LLM's draft interpretation visible and the "Use mine /
   Change it" pair of buttons rendered.
5. Click "Change it"; amend the draft; submit.
6. Confirm the audit-readiness panel's "LLM interpretations" row shows
   "1 accepted" (or "1 amended").
7. Open the runtime tool plan; confirm the LLM transform's prompt
   template now contains the user-amended interpretation string.
8. Hit "Run"; confirm the runtime Landscape `calls` row for the LLM
   transform hashes the resolved prompt string, and an
   `explain(run_id, token_id)` trace can walk the prompt string back
   to the `interpretation_events` row via the
   `composition_states.nodes` JSON.
9. Start a SECOND session; type the same hero prompt; opt out of
   interpretation review for the session ("don't ask me again this
   session"). Confirm an `interpretation_events` row with
   `choice='opted_out'` and
   `interpretation_source='auto_interpreted_opt_out'` is written by
   `record_session_interpretation_opt_out`, and the LLM proceeds to
   generate the pipeline without surfacing "cool". Confirm the
   audit-readiness panel reflects "interpretations: opted-out
   (session)".

Verification is complete when (a) all Pytest backend suites pass,
(b) all Vitest frontend suites pass, (c) the Landscape spot-check
fixture for every recording call asserts the six required fields, the
LLM-provenance columns, and `interpretation_source` are all present
and the FK resolves, (d) staging smoke passes end-to-end on the
canonical hero prompt, and (e) the opt-out path is recorded as its own
`interpretation_events` row with `choice='opted_out'` and the correct
`interpretation_source` value.

## File structure (cross-cutting)

```text
docs/composer/ux-redesign-2026-05/
  18-phase-5b-surface-llm-interpretation.md                     THIS FILE
  18a-phase-5b-backend.md                                       SIBLING (backend tasks)
  18b-phase-5b-frontend.md                                      SIBLING (frontend tasks)
```

Per-file change manifests are in 18a and 18b respectively.

## Risks

The Phase 5b risk register supersets design-spec §Risks with
implementation-specific rows:

| Risk | Mitigation |
|---|---|
| **Users feel nagged by interpretation prompts.** Design-spec §Risks identifies this as the headline UX risk. | Heuristic threshold lives in the composer-skill markdown (not code), so it can be tuned without a deploy. Bias toward false positives at launch (per design spec). Offer the per-session "stop asking" toggle from day one. Defer cross-session preference persistence to Phase 8. |
| **The LLM hallucinates the "interpretation" itself** — i.e., produces a draft that doesn't actually describe how it intends to operationalise the user's term. | Tool-argument validation at the boundary cannot detect this (it's a semantic, not structural, defect). Mitigation is twofold: (a) the user's review IS the integrity gate by design — they accept or amend the draft, so a hallucinated draft is the very thing the feature catches; (b) record the LLM's draft verbatim in `llm_draft`, separate from `accepted_value`, so forensic analysis can compare drafts to acceptances and surface a "LLM frequently produces drafts the user heavily amends" pattern as a future telemetry signal. |
| **Multi-turn interpretation editing** (user wants to amend a second time). | MVP: first user-submitted value is final. A re-surface of the same term in the same composition produces a new event (preserves history). Document this as a known limitation; track second-pass UX as a Phase 5b-followup observation. |
| **Mode discrepancy between guided and freeform.** | Both modes use the *same* backend tool surface (`request_interpretation_review`) and the *same* event-recording semantics. The surfacing UI differs by design (turn widget in guided, inline message in freeform). The Vitest integration test in 18b-Task-6 exercises both UI paths against a single backend fixture to guarantee parity. |
| **Audit trail becomes noisy with small acceptance events.** | Design-spec §Risks accepts this cost. Implementation does NOT add throttling. The audit-panel row aggregates ("N accepted, M amended, K opted-out") so the operator sees a count, not a flood. |
| **Surfacing heuristic fires AFTER the LLM has already baked the interpretation into the prompt template.** A naive implementation that calls `set_pipeline` first and then realises "oh wait, 'cool' is subjective" would defeat the feature. | The composer-skill prompt MUST instruct the LLM to surface the interpretation BEFORE binding the prompt template. The tool-handler for `request_interpretation_review` therefore takes `affected_node_id` for an LLM transform that already exists in the composition state. If the LLM has not yet created the LLM transform, the tool returns ARG_ERROR ("call this BEFORE wiring the LLM transform; you must already have the node in the composition state"). The frontend integration test asserts this ordering: surfacing comes before the LLM transform's prompt template is finalised. |
| **The user's amended interpretation contains injection-style payloads.** A user pastes `"; DROP TABLE chat_messages; --` into the amendment field. | The `accepted_value` is stored verbatim in a parameterised INSERT (SQLAlchemy Core); SQL injection is structurally impossible. The string is then interpolated into the prompt template via plain string substitution at composition time; the runtime LLM transform receives it as text. There's no SQL execution path between the user's text and any downstream system. The length cap and plain-text constraint prevent pathological payloads. |
| **Race: user resolves an interpretation event while a second compose turn is mutating composition state.** | The route handler acquires the session write lock (`_session_write_lock` per `service.py`) for the entire resolve transaction (insert event + patch state + advance version). Concurrent compose loop turns serialize on the same lock. The compose loop also reads `interpretation_events.status` for the current pipeline state; if any are pending, the LLM is told via the system prompt that the user still has pending interpretations to resolve, so a polite compose-loop turn cannot bind the prompt template ahead of the user. |
| **Failure path: the user closes the tab between the LLM tool-call and resolving the interpretation.** | The pending event row persists; on next session reload, the frontend re-renders the pending review affordance from the `list_interpretation_events(status='pending')` call. State is durable in the session audit DB, not in frontend memory. |
| **Phase 5a's `inline_blob` flow and Phase 5b's interpretation flow collide on the same hero example.** | The integration plan is explicit: 5a fires first (source creation), 5b fires after (transform-prompt interpretation). The ordering is enforced by the composer-skill prompt and by the heuristic that the interpretation tool requires an `affected_node_id` of an already-existing LLM transform. The Vitest integration test in 18b-Task-6 covers both flows in one assertion sequence. |

**Downstream prompt-injection risk (Phase 8 follow-up).** User-amended `accepted_value` content is interpolated into a runtime LLM transform's prompt template. The runtime LLM has no structural way to distinguish composer-user-controlled text from pipeline-data-controlled text — both arrive as interpolated strings. Phase 5b ships with content validation (Decision D above: rejection of `{{`/`}}`, control characters, multi-line input, and placeholder-position enforcement), which prevents the most direct injection vectors. Defence-in-depth — sentinels in the prompt template, explicit role-isolation between user-provided and system-provided prompt sections — is named as a Phase 8 polish follow-up. See [20-phase-8-polish-and-telemetry.md](20-phase-8-polish-and-telemetry.md) for the Phase 8 scope.

## Review history

| Date | Reviewer | Verdict | Notes |
|------|----------|---------|---|

(Empty — to be populated as the plan moves through review.)

---

## Open questions raised by this plan

None at the time of writing. If implementation surfaces ambiguities,
add rows to the table here and surface to the operator before
committing to an interpretation.
