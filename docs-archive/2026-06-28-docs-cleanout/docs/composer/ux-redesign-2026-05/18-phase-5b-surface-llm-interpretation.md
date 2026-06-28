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

## Worktree

**Branch:** `feat/composer-phase-5-chat-data-entry`
**Worktree path:** `/home/john/elspeth/.worktrees/composer-phase-5-chat-data-entry/`
**Shared with:** the entire Phase 5 umbrella (17-, 18-, 18a-, 18b-). Phase 5a and Phase 5b ship as a coordinated PR; do NOT split into separate branches. This document is one of the four that will be implemented together on this single worktree. Shared with 17-, 18a-, 18b- (the Phase 5a plan, Phase 5b backend plan, and Phase 5b frontend plan).

### Setup (one-time)

From the main checkout at `/home/john/elspeth`:

```bash
git worktree add .worktrees/composer-phase-5-chat-data-entry -b feat/composer-phase-5-chat-data-entry
cd .worktrees/composer-phase-5-chat-data-entry
uv venv --python 3.13                       # Python 3.13 to match main; mismatched versions produce ~300 spurious tier-model violations
source .venv/bin/activate
uv pip install -e ".[dev,llm]"              # editable install bound to THIS worktree's venv, not main's
```

### Operational notes

- **uv venv discipline:** every `uv pip install` invocation in this worktree MUST be preceded by `source .venv/bin/activate` OR invoked with `--python /home/john/elspeth/.worktrees/composer-phase-5-chat-data-entry/.venv/bin/python`. Without this, `uv` resolves to main's `.venv` and clobbers it. (See `feedback_uv_venv_leak`.)
- **filigree CLI:** the bare `filigree` command rejects realpath-escaping DBs from inside a worktree. Prefer the `mcp__filigree__*` tools. If you must use the CLI, run it from the git common dir: `(cd "$(git rev-parse --git-common-dir)/.." && filigree <verb>)`.
- **Subagent dispatch from this worktree:** subagents inherit parent CWD silently. Prefix every dispatch prompt with: "Your CWD is `/home/john/elspeth/.worktrees/composer-phase-5-chat-data-entry/`; all file paths must be absolute." Use absolute paths everywhere. (See `feedback_subagents_cant_use_worktrees`.)
- **Composer-skill edits stay on main:** the `src/elspeth/web/composer/skills/pipeline_composer.md` file is read by the live `elspeth-web.service` from main, not from any worktree. Skill-prompt edits in this phase (e.g. 5a Task 8 nudge, 5b Task 8 nudge) must be applied on main and the service restarted, per `feedback_skip_worktree_for_skill_and_config_edits`. Land the rest of the work in the worktree as normal.

### Coordination during implementation

- All four plan docs ship one commit history. The order is: 17- (Phase 5a) lands first; then 18a- (Phase 5b backend); then 18b- (Phase 5b frontend). 18- (overview) carries no code changes — its amendments land alongside whichever backend doc they cross-reference.
- The two-DB deletion requirement (session DB + Landscape audit.db) is operator-visible — surface it in the PR description so the operator can run the deletion before deploy.

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
   row's required fields), (l) `interpretation_source` (structural
   discriminant — `user_approved` | `auto_interpreted_opt_out` |
   `auto_interpreted_no_surfaces` — written by the service path, not
   the LLM), and (m) `resolved_prompt_template_hash` (SHA-256 of the
   accepted prompt-template string, populated at resolve time, enabling
   hash-anchored cross-DB linkage to the runtime Landscape `calls`
   table — see §"Hash-anchored cross-DB linkage"). None of the original
   six, nor the new six, are first-class columns on any existing table.
   Inventing them as JSON sub-fields of an existing event payload
   sacrifices query ergonomics, audit-tooling readability, and the
   spec's stated intent that interpretation events be "discrete events."

### Verdict (c) implications

Phase 5b's backend scope is:

- A **new `interpretation_events_table`** in `web/sessions/models.py`,
  with first-class columns for the six required fields plus: a closed
  enum on the choice discriminant (`accepted_as_drafted` /
  `amended` / `opted_out`); LLM-provenance columns (`model_identifier`,
  `model_version`, `provider`, `composer_skill_hash`, `arguments_hash`)
  that bind the row to the specific LLM call that produced the draft and
  make the row replayable across model upgrades; a
  `resolved_prompt_template_hash` column (SHA-256 of the accepted
  prompt-template string, populated at resolve time, enabling hash-anchored
  cross-DB linkage to the runtime Landscape `calls` table — see
  §"Hash-anchored cross-DB linkage" below); and a structural
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

### Skill-markdown archival

`composer_skill_hash` is recorded on every `interpretation_events` row as the
SHA-256 of the `pipeline_composer.md` content the composer LLM actually
received. Three reviewers independently identified a durable-archival gap: the
hash chains to a markdown file that operators can edit and the service reads
via an `@lru_cache`-loaded module (per project memory
`project_composer_harness_state`). If the file is edited, the service is
restarted, and the cache is invalidated, the historical hash points to content
that no longer exists in the filesystem.

**Required resolution (owned by 18a-):** A sibling `skill_markdown_history`
table stores `(hash, filename, content_text, first_seen_at)` tuples. The
compose loop upserts a row the first time it encounters a given hash value.
**The hash MUST be computed from the in-memory string the LLM actually
receives** — specifically, from the `@lru_cache`-memoised string returned by
the skill loader — not by re-reading the file from disk at audit-write time.
This ensures the hash and the archived content are always derived from the same
bytes. The schema and writer-path for `skill_markdown_history` are specified
in 18a- (Task 2 schema additions, Task 5 compose-loop hook).

With this table in place, any future auditor can look up the full skill
content for any `composer_skill_hash` value they encounter in the
`interpretation_events` table, even after the markdown file has been
subsequently edited or the service restarted. The archival is upsert-on-first-
reference so repeated compose runs using the same unmodified skill incur no
write overhead.

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

### Demo readiness

The canonical hero prompt — "create a list of 5 government web pages and
use an LLM to rate how cool they are" (per project memory
`project_composer_canonical_test_case`) — exercises both phases
sequentially in a single composition:

1. Phase 5a's inline-source path fires first, surfacing 5 URLs for review.
2. Phase 5b's interpretation-review affordance fires next, presenting the
   LLM's draft interpretation of "cool" for user acceptance or amendment.

**Design intent for the disambiguation predicate:** the hero prompt
produces a clean LLM-generated list of 5 URLs with an unambiguous row
count — the Phase 5a disambiguation widget (introduced in
[17-phase-5a-dynamic-source-from-chat.md](17-phase-5a-dynamic-source-from-chat.md))
MUST NOT fire for this case. If it does fire, the disambiguation predicate
is too aggressive and must be tightened before the demo. The staging-smoke
verification in 5a-Task 9 is the gate for this. The hero prompt must reach
Phase 5b's interpretation step without the user having to first resolve a
disambiguation widget, or the two-phase demo sequence breaks.

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
audits how the pipeline arrived at its current shape.

### Hash-anchored cross-DB linkage (Option A — selected)

A review panel (Audit-Arch MAJOR-6) identified that the runtime Landscape's
`calls` table carries a `request_hash` (SHA-256 over the full LLM request
envelope), which is NOT comparable to the session DB's
`interpretation_events.arguments_hash` (SHA-256 over the row's required
fields). The two hashes are computed over different inputs; a naïve equality
join would always fail. The structural join is via `composition_states.nodes`
JSON — the resolved prompt string is embedded in the composition state, and
the runtime hashes whatever prompt the LLM transform receives at execution
time.

**Option A is selected:** Phase 5b introduces a `resolved_prompt_template_hash`
column on BOTH:

1. `interpretation_events_table` (session audit DB) — populated at resolve
   time as the SHA-256 of the accepted prompt-template string (the verbatim
   string that will be embedded into `composition_states.nodes`).
2. The runtime Landscape `calls` table — populated at execution time as the
   SHA-256 of the resolved prompt string the LLM transform actually receives.

Both sides compute the hash using the same `CANONICAL_VERSION = "sha256-rfc8785-v1"`
scheme from `contracts/hashing.py`. Because the resolved prompt string is
deterministically embedded at composition time and read back verbatim at
execution time, the two hashes MUST be equal for any uncorrupted run. An
inequality indicates tampering or a composition-to-execution coherence failure.

**18a- Task 2 lands the column** on `interpretation_events_table`. The runtime
`calls` table column is also specified there (alongside the Landscape schema
note); 18a- Task 9 adds a verification assertion that the hash values match
across the two DBs for the hero-prompt integration run.

With this column in place, the attributability test holds with a hash-anchored
join: `explain(recorder, run_id, token_id)` for any output traces back through
the runtime `calls` row (via `resolved_prompt_template_hash`) → the
`interpretation_events` row (matched by the same hash) → the user who accepted
it. The structural join via `composition_states.nodes` remains authoritative
as the backup path.

## Tech Stack

- Python 3.13, SQLAlchemy Core (session audit DB), pydantic (wire
  schemas + tool argument validation).
- React + Zustand + Vitest + testing-library for the frontend.
- The composer skill (Markdown prompt) for the LLM-side heuristic
  nudge.
- No new runtime dependencies; no new MCP-server surface.
- One Landscape schema change: a `resolved_prompt_template_hash` column
  added to the runtime `calls` table (per §"Hash-anchored cross-DB
  linkage" Option A; specified in 18a-).

## Sibling plans

| Plan | Status | Relationship |
|------|--------|---|
| [17-phase-5a-dynamic-source-from-chat.md](17-phase-5a-dynamic-source-from-chat.md) | Implementation in progress / ready to ship | **Prerequisite**. Phase 5a establishes the canonical hero example (`"create a list of 5 government web pages and use an LLM to rate how cool they are"`) end-to-end through inline-source-from-chat. Phase 5b layers the *second* feature onto the same hero example — the "cool" interpretation. The plans must integrate cleanly: Phase 5b's frontend turn widget must coexist in the same `ChatPanel.tsx` dispatch as Phase 5a's `InlineSourceCreatedTurn`. |
| [04-first-run-tutorial.md](04-first-run-tutorial.md) (Phase 4) | Dependent | Phase 4's hello-world tutorial *consumes* Phase 5b. The tutorial's third beat is the interpretation review of "cool"; until Phase 5b ships, the tutorial cannot teach that affordance. Phase 5b must ship before Phase 4 is plannable. |
| [14-phase-2-audit-readiness-panel.md](14-phase-2-audit-readiness-panel.md) | Already shipped (or in flight) | Phase 5b adds a **new conditional row** to the audit-readiness panel: "LLM interpretations: pending / accepted / none". If Phase 2 has shipped at planning time, Phase 5b's Task 18b-7 wires the row in. If not, the row is deferred to a Phase-2 followup and noted on the umbrella PR. |
| [16-phase-7-catalog-reshape.md](16-phase-7-catalog-reshape.md) | Independent | No interaction. |

## Cross-phase provenance consistency

Phase 5a and Phase 5b apply the same five-column LLM-provenance binding to
all LLM-authored content:

| Column | Meaning |
|--------|---------|
| `model_identifier` | e.g., `"anthropic/claude-opus-4-7"` |
| `model_version` | provider-reported version string |
| `provider` | e.g., `"anthropic"`, `"openai"` |
| `composer_skill_hash` | SHA-256 of `pipeline_composer.md` (the skill markdown the LLM received) |
| `arguments_hash` / `content_hash` | `sha256-rfc8785-v1` canonical hash over the auditable fields |

Phase 5a writes these five columns to the `blobs_table` for blobs with
`creation_modality` in `{'llm_generated', 'disambiguated',
'llm_generated_then_amended'}` (per the 5a plan). Phase 5b writes the same
five columns to every `interpretation_events` row. The symmetry is deliberate:
any audit-trail query asking "what LLM, running what skill, produced this
content?" can be answered identically from both the blob and the interpretation
event, making the audit story coherent across phases.

The `CANONICAL_VERSION` constant (`"sha256-rfc8785-v1"`, defined at
`src/elspeth/contracts/hashing.py` line 25) is the single source of truth for
the hash scheme used by both phases. Neither phase introduces a new hash
algorithm; both inherit the existing `stable_hash()` / `canonical_json()`
contract from `contracts/hashing.py`.

## Scope boundaries

**In scope (Phase 5b):**

- New `interpretation_events_table` in the session audit DB with the
  six required fields; LLM-provenance columns (`model_identifier`,
  `model_version`, `provider`, `composer_skill_hash`, `arguments_hash`);
  a `resolved_prompt_template_hash` column for hash-anchored cross-DB
  linkage to the runtime Landscape `calls` table (see §"Hash-anchored
  cross-DB linkage"); a structural `interpretation_source` discriminant
  (closed enum: `user_approved` | `auto_interpreted_opt_out` |
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
  records the per-session "Stop reviewing interpretations this session" decision as a row in
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
- **The "Stop reviewing interpretations this session" toggle's per-session
  persistence across sessions.** The opt-out is session-scoped. Persisting
  it across sessions (e.g., as a user preference) is a Phase 8 polish item.
- **Rate-limit configuration and runtime tuning.** The server-side
  rate caps (3 surfacings per term, 10 per session-day) are shipped
  fixed in Phase 5b (implemented in 18a-). Making them config-driven
  and operator-tunable is a Phase 8 task once usage data exists.
- **Runtime Landscape extension for a named FK backlink.** Phase 5b
  introduces `resolved_prompt_template_hash` on both
  `interpretation_events_table` and the runtime Landscape `calls` table,
  enabling a hash-anchored equality join between the two DBs (see
  §"Hash-anchored cross-DB linkage"). What remains deferred to Phase 11
  is a named `interpretation_event_id` foreign-key backlink on the
  runtime `calls` table row — a direct UUID pointer that would let an
  auditor skip the hash comparison. The hash join is the MVP; the FK
  shortcut is a Phase 11 audit-tooling ergonomic improvement.
- **Per-row redaction of `user_term` / `llm_draft` / `accepted_value`
  content.** Retention follows session-level deletion until then; Phase
  11 audit-tooling will introduce a redaction event class parallel to
  the audit-pack discipline.
- **PII retention.** The `user_term`, `llm_draft`, and `accepted_value`
  columns store free text with only the credential-shape prefilter applied
  at the route boundary. Email addresses, phone numbers, names, postal
  addresses, and other PII entered as interpretation terms or drafts are
  stored verbatim in the session audit DB. Retention is governed only by
  session-level deletion; no per-field redaction runs automatically.
  Operators deploying Phase 5b must communicate this to users. Phase 11
  will add broader automatic redaction for these fields; this deferral is
  deliberate and documented here so operators know what they are shipping.
- **Hash-chaining of `interpretation_events` rows for tamper
  detection.** The new event class is the first session-DB table whose
  contents materially affect runtime behaviour (the resolved prompt
  template). Tracked as a Phase 11 audit-tooling enhancement.

## Opt-out semantics

Phase 5b ships a **binary opt-out**: the user either reviews all surfaced
interpretations for the session, or invokes "Stop reviewing interpretations
this session" to skip all subsequent reviews. The opt-out is session-scoped;
there is no cross-session persistence in Phase 5b.

The toggle label throughout this plan and in 18b-'s frontend implementation is
**"Stop reviewing interpretations this session"** — this phrasing makes the
session scope explicit and avoids the misleading implication that the feature
itself is being disabled. See 18b- for the UI rename.

A future Phase 8 may add a third state — "auto-accept future interpretations
this session" — that commits the LLM's draft without presenting the review
widget. Phase 5b does not implement this third state; the binary is the MVP.
If Phase 8 adds the third state, the `interpretation_source` closed enum will
require a governance extension (new value `auto_accepted_by_user_preference`)
following the NO SILENT EXTENSION ceremony documented at `models.py`
lines `274-289`.

---

## Trust-tier check (mandatory before any data-handling work)

Phase 5b touches **three distinct trust boundaries** and the discipline
differs at each. This section is the canonical reference; the per-task
sections in 18a and 18b refer back to it.

| Boundary | Direction | Trust tier | Discipline |
|---|---|---|---|
| LLM → composer service | Tool-call arguments crossing into `request_interpretation_review` | **Tier 3 (external)** | Pydantic-validated at the tool boundary. Type-and-range coerce where safe (`user_term` is a `str` with a length cap; `llm_draft` is a `str` with a length cap; `affected_node_id` is validated to exist in the current composition state). Reject (ARG_ERROR) anything that doesn't conform. **Never persist the LLM's draft without surfacing it for user review** — that's the entire feature. |
| User → backend route | User's "Use mine / Change it" body crossing into `/resolve` | **Tier 3 (external)** | Pydantic-validated at the route boundary. The user's amended text is a `str` with a length cap (8 KiB after stripping) and strict content validation: reject `{{` or `}}` (template-injection guard), control characters (U+0000–U+001F except tab), and multi-line input. A placeholder-position check further requires that any `{{interpretation:…}}` token in the downstream prompt template sits in a value-position, not inside a `system:` / `role:` / `instructions:` section header. The choice discriminant is a closed enum (`accepted_as_drafted` / `amended` / `opted_out`); reject anything else. |
| Session audit DB → composer + frontend | Reading back a persisted interpretation event | **Tier 1 (ours)** | Read into a frozen dataclass (`InterpretationEventRecord`); crash on any anomaly (NULL in a NOT-NULL column, invalid enum value, dangling FK). No coercion. No `.get()`-with-default. The dataclass carries all columns: the six required fields, `model_identifier` / `model_version` / `provider` / `composer_skill_hash` / `arguments_hash`, `resolved_prompt_template_hash`, and `interpretation_source`. |

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
   `composer_skill_hash`, `arguments_hash`),
   `resolved_prompt_template_hash`, and `interpretation_source`
   is present, of the expected type, and matches the expected value.
4. Verifies the FK to `composition_states` resolves (i.e., the
   pipeline-state reference is real, not dangling).
5. For resolved rows: asserts `resolved_prompt_template_hash` equals the
   SHA-256 of the runtime `calls` row's resolved prompt string (the
   hash-equality assertion from §"Hash-anchored cross-DB linkage";
   see 18a-Task 9).

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
9. Start a SECOND session; type the same hero prompt; click "Stop
   reviewing interpretations this session". Confirm an `interpretation_events` row with
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

> This overview doc lists only plan-doc siblings (no code-file manifest). The per-file code-file manifests in 18a and 18b are implemented in the shared worktree at `/home/john/elspeth/.worktrees/composer-phase-5-chat-data-entry/`, which mirrors main's tree structure. See the Worktree section above.

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
| **Users feel nagged by interpretation prompts.** Design-spec §Risks identifies this as the headline UX risk. | Heuristic threshold lives in the composer-skill markdown (not code), so it can be tuned without a deploy. Bias toward false positives at launch (per design spec). Offer the per-session "Stop reviewing interpretations this session" toggle from day one. Defer cross-session preference persistence to Phase 8. |
| **The LLM hallucinates the "interpretation" itself** — i.e., produces a draft that doesn't actually describe how it intends to operationalise the user's term. | Tool-argument validation at the boundary cannot detect this (it's a semantic, not structural, defect). Mitigation is twofold: (a) the user's review IS the integrity gate by design — they accept or amend the draft, so a hallucinated draft is the very thing the feature catches; (b) record the LLM's draft verbatim in `llm_draft`, separate from `accepted_value`, so forensic analysis can compare drafts to acceptances and surface a "LLM frequently produces drafts the user heavily amends" pattern as a future telemetry signal. |
| **Multi-turn interpretation editing** (user wants to amend a second time). | MVP: first user-submitted value is final. A re-surface of the same term in the same composition produces a new event (preserves history). Document this as a known limitation; track second-pass UX as a Phase 5b-followup observation. |
| **Mode discrepancy between guided and freeform.** | Both modes use the *same* backend tool surface (`request_interpretation_review`) and the *same* event-recording semantics. The surfacing UI differs by design (turn widget in guided, inline message in freeform). The Vitest integration test in 18b-Task-6 exercises both UI paths against a single backend fixture to guarantee parity. |
| **Audit trail becomes noisy with small acceptance events.** | Design-spec §Risks accepts this cost. Implementation does NOT add throttling. The audit-panel row aggregates ("N accepted, M amended, K opted-out") so the operator sees a count, not a flood. |
| **Surfacing heuristic fires AFTER the LLM has already baked the interpretation into the prompt template.** A naive implementation that calls `set_pipeline` first and then realises "oh wait, 'cool' is subjective" would defeat the feature. | The composer-skill prompt MUST instruct the LLM to surface the interpretation BEFORE binding the prompt template. The tool-handler for `request_interpretation_review` therefore takes `affected_node_id` for an LLM transform that already exists in the composition state. If the LLM has not yet created the LLM transform, the tool returns ARG_ERROR ("call this BEFORE wiring the LLM transform; you must already have the node in the composition state"). The frontend integration test asserts this ordering: surfacing comes before the LLM transform's prompt template is finalised. |
| **The user's amended interpretation contains injection-style payloads.** A user pastes `"; DROP TABLE chat_messages; --` into the amendment field. | The `accepted_value` is stored verbatim in a parameterised INSERT (SQLAlchemy Core); SQL injection is structurally impossible. The string is then interpolated into the prompt template via plain string substitution at composition time; the runtime LLM transform receives it as text. There's no SQL execution path between the user's text and any downstream system. The length cap and plain-text constraint prevent pathological payloads. |
| **Race: user resolves an interpretation event while a second compose turn is mutating composition state.** | The route handler acquires the session write lock (`_session_write_lock` per `service.py`) for the entire resolve transaction (insert event + patch state + advance version). Concurrent compose loop turns serialize on the same lock. The compose loop also reads `interpretation_events.status` for the current pipeline state; if any are pending, the LLM is told via the system prompt that the user still has pending interpretations to resolve, so a polite compose-loop turn cannot bind the prompt template ahead of the user. |
| **Failure path: the user closes the tab between the LLM tool-call and resolving the interpretation.** | The pending event row persists; on next session reload, the frontend re-renders the pending review affordance from the `list_interpretation_events(status='pending')` call. State is durable in the session audit DB, not in frontend memory. |
| **Phase 5a's `inline_blob` flow and Phase 5b's interpretation flow collide on the same hero example.** | The integration plan is explicit: 5a fires first (source creation), 5b fires after (transform-prompt interpretation). The ordering is enforced by the composer-skill prompt and by the heuristic that the interpretation tool requires an `affected_node_id` of an already-existing LLM transform. The Vitest integration test in 18b-Task-6 covers both flows in one assertion sequence. |
| **20% of sessions may produce pipelines with unresolved `{{interpretation:…}}` placeholders** (Task 0's gate accepts ≥8/10 LLM probe success; the 20% tail is an acknowledged risk). | When the runtime detects an unresolved `{{interpretation:<term>}}` placeholder in an LLM-transform prompt template, the execute path MUST surface a user-actionable banner — "pipeline contains unresolved interpretation placeholders — please re-compose" — rather than running with literal placeholder text. This prevents a silent malfunction where the LLM transform receives the verbatim placeholder string instead of the user's accepted interpretation. The runtime detection implementation is specified in 18a-Task 5 (F-17 unresolved-placeholder detection sub-task and F-21 runtime telemetry signal sub-task). The user path is: see the banner → return to the composer session → complete or dismiss the pending interpretation review → re-execute. |

**Downstream prompt-injection risk (Phase 8 follow-up).** User-amended `accepted_value` content is interpolated into a runtime LLM transform's prompt template. The runtime LLM has no structural way to distinguish composer-user-controlled text from pipeline-data-controlled text — both arrive as interpolated strings. Phase 5b ships with content validation (Decision D above: rejection of `{{`/`}}`, control characters, multi-line input, and placeholder-position enforcement), which prevents the most direct injection vectors. Defence-in-depth — sentinels in the prompt template, explicit role-isolation between user-provided and system-provided prompt sections — is named as a Phase 8 polish follow-up. See [20-phase-8-polish-and-telemetry.md](20-phase-8-polish-and-telemetry.md) for the Phase 8 scope.

## Phase 9 migration debt

Phase 5b adds four interdependent DDL objects to the session audit DB that
the Phase 9 migration runner must reproduce in dependency order:

1. **`interpretation_events_table`** — the new event table itself, with its
   six required-by-spec columns, LLM-provenance columns, closed-enum CHECK
   constraints, the composite FK to `composition_states`, and the partial
   unique index on `(session_id, tool_call_id)` where `choice='pending'`.

2. **`interpretation_review_disabled` column on `sessions_table`** — a
   `BOOLEAN NOT NULL DEFAULT false` column on the existing `sessions` table.
   SQLite does not support `ADD COLUMN … DEFAULT` for non-trivial expressions
   via `ALTER TABLE`; the migration runner must handle this via
   table-recreation if the staging DB predates Phase 5b.

3. **`interpretation_resolve` value on the `composition_states.provenance`
   CHECK constraint** — the closed-enum extension documented at `models.py`
   lines `274-289`. SQLite CHECK constraints live in the `CREATE TABLE`
   statement and cannot be altered in-place; the migration runner must
   recreate `composition_states_table` with the extended constraint, preserving
   all existing rows. This step MUST follow any step that relies on
   `composition_states` rows being readable.

4. **The append-only `BEFORE UPDATE` trigger on `interpretation_events`** —
   rejects mutations to settled fields after `resolved_at` is set. Must be
   installed after the table exists.

**Recommended order:** `sessions` column addition → `composition_states` table
recreation → `interpretation_events` table creation → trigger installation.

18a- will deposit a "Phase 9 migration notes" block adjacent to the schema
definitions for the Phase 9 implementer. The Phase 9 migration runner is
SQLite-only as of 2026-05-16 (per project memory `project_phase9_sqlite_only`);
this table-recreation pattern is consistent with its current design.

---

## Review history

| Date | Reviewer | Verdict | Notes |
|------|----------|---------|---|
| 2026-05-18 | 9-reviewer panel (audit-architecture, embedded-database, LLM-safety, threat-analyst, plan-review × 4, UX-critic) | GO-WITH-AMENDMENTS (closes pre-existing review panel cycle) | F-1 hash-anchored cross-DB linkage (Option A); F-2 skill_markdown_history; F-3 cross-phase provenance; F-4 demo readiness; F-5 rate-limit deferral rename; F-6 Phase 9 migration debt; F-7 20% LLM-gate failure case; F-8 PII retention deferral; F-9 opt-out semantics + 'Stop reviewing interpretations this session' rename |
| 2026-05-18 | reviewer pass (complex-reviewer) | NO-GO → GO once 18a- F-1 closure lands (`resolved_prompt_template_hash` column on both interpretation_events_table and runtime Landscape calls_table, plus Task 9 hash-equality assertion) | Verified F-2 through F-9 landed correctly; F-1 prose correct but forward-references to 18a- Task 2/Task 9 were unresolved at first check — closure dispatched |
| 2026-05-18 | Independent contradiction review | APPLIED | Stale cross-reference corrected: line-683 risk table row "runtime detection implementation is specified in 18a-Task 9" updated to 18a-Task 5 (F-17 detection sub-task and F-21 telemetry-signal sub-task). Task 9 covers hash-equality verification, not placeholder detection. |
| 2026-05-18 | Plan amendment | APPLIED | Added shared-worktree section: `feat/composer-phase-5-chat-data-entry` at `.worktrees/composer-phase-5-chat-data-entry`; Phase 5a + Phase 5b ship as coordinated PR on one branch. Added worktree-root context note to File structure section. |

---

## Open questions raised by this plan

None at the time of writing. If implementation surfaces ambiguities,
add rows to the table here and surface to the operator before
committing to an interpretation.
