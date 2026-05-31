# Composer reset + background debrief, and the two root-cause fixes behind it

**Date:** 2026-05-31
**Status:** Design — awaiting operator review
**Scope owner:** composer (web) subsystem

## Motivation

A real staging session (`2e6d5e3e-1848-4da3-80bd-725b61a4f459`, user `dta_user`,
title "CSV Web Downloader Colour Analysis") was reported as "slow / didn't
seem to converge." Audit-trail analysis (`data/sessions.db`) showed the
conversion *did* succeed structurally but exhibited two pathologies:

1. A **~2-minute single-call stall** (seq 11 → seq 12, 02:28:05 → 02:30:08).
   It followed a recoverable validation rejection — the model submitted a CSV
   source schema as `mode: observed` *with* an explicit `fields` array. The
   rejection response shipped the **entire** `csv` plugin schema back
   (~5 KB: description + `json_schema` + `knob_schema` + 13 composer_hints),
   ballooning context to ~40 K tokens, and the cheap composer model
   (`gpt-5.4-mini`) took ~120 s to regenerate the corrected pipeline. The
   schema dump was *unnecessary*: the validation error message
   ("Observed schemas cannot have explicit field definitions. Use
   guaranteed_fields/required_fields instead.") was already a complete fix.

2. A **turn-13 dead-end on a wire-visible value.** The final composition state
   (v7) was still invalid on one rule: `web_scrape.http.abuse_contact =
   compliance@example.com`, rejected because `example.com` is RFC-2606/6761
   reserved and would ship a fabricated abuse-contact identity. The model
   correctly *refused to invent* a real email and asked the operator to supply
   one — but only after 13 turns of work. The skill already instructs the model
   to collect wire-visible values *before building* (skill lines 173-175); the
   cheap model ignored the prose instruction.

The operator requested a "hard reset and start over" button that also captures
"everything that was wrong and everything that was right." Requirements were
elicited by interviewing the *actual* user of the composer toolset — an LLM —
across three capability tiers (Haiku ≈ the live cheap composer, Sonnet, Opus)
to get a representative spread, since the toolset is LLM-facing.

**The honest framing (operator-approved scope "both, root-cause first"):** the
reset button is largely *compensating* for the two root causes above. We fix
those first, then build the reset/debrief as a clean escape hatch on top.

## Non-goals

- No human-facing "report card" / postmortem panel. The debrief is **background
  plumbing for the LLM and the audit trail**, not a user-facing surface
  (operator directive 2026-05-31). The tutorial especially must keep this
  invisible — first-run hello-world just recovers and continues.
- No removal of the failure-time schema augment. It exists for a real reason
  (composer session `47cfbb5e`: agents almost never converged without
  context-specific guidance). We make it *delivery-aware*, not absent.
- No `delete_session`. The prior session is archived, never hard-deleted.
- No new LLM model/provider wiring. Reuse the existing compose loop, advisor
  path, and audit writers.

## Settled cross-cutting decisions

These apply across the stages and are load-bearing:

- **D1 — Never re-seed the 5 KB schema.** The debrief and the suppressed-dump
  path must never carry the full `PluginSchemaInfo.model_dump()` payload;
  re-introducing it re-causes the stall.
- **D2 — Lifeline changes form, never disappears.** When the schema dump is
  suppressed, the response **must** include an explicit, copy-pasteable pointer
  naming the exact tool call to retrieve the full data (see Stage 1). A terse
  error with no convergence path would regress to the pre-augment
  non-convergence problem.
- **D3 — `blocked_pending_human` ≠ `rejected`.** A correct refusal to fabricate
  a wire-visible value (e.g. `abuse_contact`) is *right behaviour* and must
  never be filed as a model error. Miscategorising it would train the next
  attempt to fabricate its way past the validator — directly attacking
  ELSPETH's anti-fabrication doctrine.
- **D4 — Audit always records; synthesis is flagged.** The debrief is recorded
  in the audit trail regardless of human visibility. Deterministic facts are
  asserted; LLM-synthesized narrative/classification is tagged
  `synthesized: true` so an auditor never mistakes narrative for a recorded
  fact ("no inference — if it's not recorded, it didn't happen").

## Stage 1 — Error-aware failure augmentation (root cause #1)

### Current behaviour
`_augment_with_plugin_schemas` (`src/elspeth/web/composer/tools/_dispatch.py:329`)
fires for tools whose declaration sets `augments_on_failure=True`. On any failed
result it calls `build_plugin_schemas_for_failure` (`tools/_common.py:821`),
which scans `result.validation.errors` for messages matching
`_INVALID_OPTIONS_PLUGIN_RE` (`"Invalid options for (source|transform|sink)
'<plugin>'"`, `_common.py:816`) and inlines the full schema for every matched
plugin. The gate is keyed on **tool name only** — it cannot distinguish
"model has never seen this schema" (dump helps) from "model already has the
schema and made a semantic slip" (dump is pure bloat that re-enters context on
every subsequent turn).

### Proposed behaviour
Make the gate **delivery-aware**:

- Track delivered `(kind, plugin)` schema pairs per session in a compose-loop
  carrier (the loop already tracks per-session state such as
  `repair_turns_used` in `_compose_loop_carriers.py`). Populate it whenever
  `get_plugin_schema` is called *or* an augment fires.
- **First** option-shape rejection for a not-yet-delivered plugin → dump the
  full schema (unchanged; genuinely helpful for cold start). Record delivery.
- **Repeat** option-shape rejection for an already-delivered plugin →
  **suppress the dump** and instead return the verbatim validation error plus a
  structured, explicit pull-directive (D2), e.g.:

  ```
  Full schema already provided earlier this session. For the full schema call
  get_plugin_schema(kind="source", plugin="csv"); for targeted help call
  get_plugin_assistance(kind="source", plugin="csv", issue_code="<code>").
  ```

  `get_plugin_assistance(issue_code=…)` (`tools/generation.py:470`) and
  `explain_validation_error` (`:426`) already exist for exactly this.

### Scope decision (operator-approved)
Keep the **first** dump full; only suppress **repeats**. Trimming the cold-start
dump size is a separate, measure-first follow-up — not part of this stage.

### Plumbing note for the plan
`_augment_with_plugin_schemas` currently takes `(result, tool_name, catalog)`.
It needs read/write access to the per-session delivered-schema set. Confirm the
cleanest injection point in `execute_tool` / `tool_batch` dispatch (the
accumulator is already threaded there for `advisor_calls_used` etc.).

### Code-comment requirement
At the suppression site, add a comment citing D2 and the `47cfbb5e` rationale,
so a future session does not "simplify" the explicit pull-directive away and
silently reintroduce non-convergence.

## Stage 2 — Wire-visible pre-flight gate, single source of truth (root cause #2)

### Current state (the constraint is encoded three times)
1. Plugin field validator (`plugins/transforms/web_scrape.py:100-108`) rejects
   placeholder values via `is_wire_visible_placeholder`
   (`contracts/wire_visible_identity.py:41`) — **reactive**, fires only after a
   value is set (i.e. at turn 13).
2. `web/composer/implicit_decisions.py:198,206` classifies
   `web_scrape.{http.abuse_contact, http.scraping_reason}` as `identity` /
   `explicit_source_required` — a **post-hoc audit label** on an already-set
   value, by **hardcoded path suffix**.
3. Skill prose (lines 173-175) — ignored by the cheap model.

None is a pre-flight gate. The same wire-visible contract is also used by
`dataverse` / `database_sink` / `azure_blob_sink` / `azure_blob_source` /
`dataverse` source (all import `reject_placeholder_value`), so the concept is
cross-plugin but the *enumeration* of which fields are operator-required lives
only in scattered hardcoded lists and prose.

### Proposed shape (operator-approved: mechanical gate + dedupe)
1. **Single declarative source.** Plugins declare their wire-visible-required
   fields once (a classmethod or field-level marker resolved through the
   existing `wire_visible_identity` contract). `implicit_decisions.py` and the
   new gate both read from it; the hardcoded path list in
   `implicit_decisions.py` is removed in favour of the declarative source. This
   generalises the gate to every `reject_placeholder_value` plugin, not just
   `web_scrape`.
2. **Pre-flight gate in the compose loop.** When a chosen plugin has
   wire-visible-required fields that are unset or placeholder-valued, the gate
   marks them **`blocked_pending_human`** (not `rejected`, per D3) and surfaces
   an up-front clarifying question through the existing
   interpretation-review / clarifying-question path — *instead of* letting the
   model fabricate a placeholder and dead-end later. The topology may still
   build; the field is simply marked blocked until the operator supplies it.

### Synergy with Stage 3
`blocked_pending_human` is the exact taxonomy the debrief needs. Implementing it
as a real state here gives Stage 3 its vocabulary directly, rather than
re-deriving "mistake vs correct refusal" heuristically.

## Stage 3 — Reset + background debrief (the requested feature)

### Reset mechanism — new-session-by-reference (not fork-with-copy, not wipe-in-place)
`fork_session` (`web/sessions/service.py:4618`) copies the prior composition
*state* forward; "start over" wants the opposite — an empty graph. So this is a
new `reset_session` operation that reuses fork's plumbing (prior-session
linkage, blob copy, read-only history reference) but seeds an **empty**
composition state (version 1).

- **Archive, never delete** the prior session (`archive_session`,
  `service.py:2181`) — immutable audit record and the debrief's only
  deterministic source. (`delete_session` is a hard-delete with no undo;
  explicitly not used.)
- **New session**, empty graph, `reset_from: <prior_session_id>` pointer.
- **Carry forward:** user-*uploaded* blobs (don't re-prompt for the CSV); the
  original intent (first human message); already-*accepted* interpretation
  decisions (offered as fast-path defaults); the debrief seed.
- **Do NOT carry forward:** half-built nodes/edges/options; pending unaccepted
  interpretation cards; LLM-authored draft blobs; the failed reasoning; and
  never the 5 KB schema (D1).

Wipe-in-place is rejected: it would orphan pending interpretation events against
a vanished state version and surface them as phantom blockers at
`_runtime_preflight` on the new composition. The new-session boundary avoids
this.

### The debrief — background only, two consumers (LLM + audit)
There is **no human-facing card** (operator directive). One synthesis pass
produces:

- **LLM seed:** a compact, structured `system`-role message
  (`writer_principal = session_fork` or a dedicated reset principal) at the
  start of the new session — because the next model has no cross-session memory;
  if it is not in context at turn 1, it does not exist.
- **Audit record:** the same structured artifact persisted to the trail (D4).

Content, split by trust:

- **Deterministic (asserted, from the audit trail):**
  - cost: `turns`, `tool_calls`, `slowest_call_s`;
  - `rejected`: validator-caught model mistakes (the `observed`+`fields`
    contradiction; the missing sink edge), each with "introduced at vN, fixed
    at vM" derived from composition-state lineage;
  - `blocked_pending_human`: correct refusals (e.g. `abuse_contact`), flagged as
    *right behaviour* plus "operator must provide X";
  - `correct_topology_skeleton`: the structurally-valid source/node/sink shape
    (carry-forward scaffolding);
  - `resolved_reviews`: accepted interpretation decisions.
- **Synthesized (LLM, one line each, `synthesized: true`):** the *why* narrative
  and the right/wrong/blocked classification.
- **Never contains (D1):** the 5 KB schema, full chat history, half-built
  topology, or confabulated root-cause guesses.

Illustrative seed shape (final field names settled in the plan):

```json
{
  "reset_from": "<prior_session_id>",
  "cost": {"turns": 13, "tool_calls": 14, "slowest_call_s": 118},
  "correct_topology_skeleton": "csv -> web_scrape -> llm -> json",
  "constraints_learned": [
    "csv source: mode:observed forbids explicit fields[]; use guaranteed_fields/required_fields",
    "web_scrape.http.abuse_contact must come from the operator — never fabricate (example.com is reserved)"
  ],
  "blocked_pending_human": [
    {"field": "web_scrape.http.abuse_contact", "reason": "no real abuse contact supplied"}
  ],
  "resolved_reviews": ["llm_prompt_template:color_analysis", "llm_model_choice:color_analysis"],
  "synthesized": true
}
```

### Carry-forward line
Constraints + correct topology skeleton + accepted decisions carry forward;
**state + reasoning do not.** `blocked_pending_human` items become attempt 2's
first clarifying questions.

### Refinements folded in (operator-approved)
- **Emit the debrief on any terminal stop, not only on reset** — as a background
  LLM seed + audit record, never a human notification. This often makes reset
  unnecessary: the operator just supplies the missing value and continues. It is
  the behaviour that would have served the original session.
  > **SCOPE UPDATE (2026-05-31 plan review):** this refinement is **split out of
  > the Stage 3 implementation plan into a separate follow-up** (it was Task 6).
  > It is the riskiest piece — it touches the hot compose-loop finalize path and
  > the tutorial-invisibility guarantee, and needs a real (non-stub) test plus a
  > code-verified hook point. Stage 3 ships the reset feature without it; the
  > follow-up owns this refinement. The design intent here is unchanged.
- **Record reset as a named, audited Landscape event** (`reset_from` → new
  session): reversible-by-reference (nothing destroyed) and a clean eval signal
  for tracking composer convergence across model/skill versions.

### Human-visible surfaces (the only ones)
1. The reset button itself.
2. Natural clarifying questions the LLM asks as a result of a
   `blocked_pending_human` item (e.g. "what abuse-contact email should I use?").

The tutorial path must keep everything else invisible.

## Post-recon corrections (2026-05-31)

Code reconnaissance after the initial design verified the citations and
**falsified several assumptions**. Recorded here so the design is truthful and
the per-stage plans build on reality.

### Stage 1 — better than assumed
The session-scoped "schemas delivered this session" tracker **already exists**:
`_schemas_loaded_by_session` (`web/composer/service.py:984`), written by
`_mark_plugin_schema_loaded` (`:2581`) on every successful `get_plugin_schema`
(`tool_batch.py:1342`), read via `_schemas_loaded_for_session` (`:2566`). Stage 1
does not build a new carrier — it post-processes the dispatched `ToolResult` in
`tool_batch` using this set. See
`docs/superpowers/plans/2026-05-31-composer-stage1-error-aware-augmentation.md`.

### Stage 2 — two corrections
- **No `blocked_pending_human` enum exists.** The existing "pending human" seam
  is `InterpretationKind` (`web/composer/composer_interpretation.py:74-85`) — a
  **closed, governance-gated** enum that models *LLM assumptions the user
  reviews*, persisted as a Tier-1 `InterpretationEventRecord` with DB check
  constraints. A missing wire-visible value is an *unset operator input*, not an
  LLM assumption — a semantic mismatch. **Decision (recommended): build a
  parallel readiness blocker** alongside `InterpretationReviewPending`
  (`web/composer/interpretation_state.py:108-115`, returned by
  `materialize_state_for_execution`, `:415-434`), reusing that return-union
  pattern, rather than adding a new `InterpretationKind`. The vocabulary token
  `blocked_pending_human` becomes this new blocker type, not an interpretation
  kind.
- **The declarative source must NOT live on `PluginAssistance`/`composer_hints`**
  — that module is explicitly advisory and non-contractual
  (`contracts/plugin_assistance.py:65-71`), while wire-visible-required is
  load-bearing for whether the pipeline may run. **Decision: a new contractual
  classmethod on the plugin base** (`plugins/infrastructure/base.py:798`,
  parallel to `get_agent_assistance`), e.g. `wire_visible_required_fields()`.
- **Duplication is 7 surfaces, not 3:** web_scrape `Field(...)` (`:82-89`),
  `_reject_empty` validator (`:100-110`), `get_agent_assistance` hint prose
  (`:544`), `implicit_decisions._category_for_node_option` (`:198`),
  `_provenance_for_path` (`:206`), `_note_for_node_option` (`:236`), skill prose.
  The dedupe must reproduce **two match styles** (suffix match in
  `_provenance_for_path`; `node.plugin` + exact nested `field_path` elsewhere)
  and preserve nested paths (`http.abuse_contact`). Regression guard:
  `tests/unit/web/sessions/test_routes.py:5856` asserts exact
  `provenance="explicit_source_required"` output — must stay green.

### Stage 2 — tutorial seeding (decided 2026-05-31)
The hello-world tutorial drives the model with `CANONICAL_SEED_PROMPT`
(`web/composer/tutorial_service.py:48,111`) — the `web_scrape` colour-analysis
hero example — then normalizes + caches the result (`_normalise_tutorial_*`,
`:242`; provenance `tutorial_normalization`). Today the model invents
`abuse_contact` → `example.com` → the rejection seen in `2e6d5e3e`.

**Decisions:**
- The tutorial **deterministically seeds both** wire-visible-required fields so
  neither surfaces during the tutorial: `http.abuse_contact = elspeth@dta.gov.au`
  and a fixed `http.scraping_reason` (e.g. "DTA composer tutorial — government
  website colour analysis"). These are **operator/deployment-identity-supplied**
  values (legitimate per the wire-visible rule and the fabrication doctrine — the
  operator supplies them, the model does not invent them), so Stage 2's gate
  passes silently and the tutorial stays smooth.
- **Scope: tutorial-only.** No deployment-wide default `abuse_contact`; real user
  pipelines still hit Stage 2's gate and are asked for a real contact.
- **Mechanism: deterministic injection during tutorial normalization**, not a
  prompt instruction — consistent with this whole effort's lesson (do not rely on
  the cheap model for wire-visible values). Inject into the `web_scrape` node's
  `http` options if unset/placeholder, in the `_normalise_tutorial_*` path.
- **Cache caveat:** the tutorial caches the canonical-prompt result
  (`tutorial_service.py:121-153`); changing the seeded values requires
  re-seeding / invalidating that cache so the cached pipeline carries the new
  values.

### Stage 3 — five corrections, several need operator sign-off
- **C2 — RESOLVED (advisor-confirmed 2026-05-31): no new chat role.** The recon
  conflated two jobs that should stay separate:
  - *Reach the next LLM attempt* → **compose-time system-context injection**, not
    a transcript row. `build_context_string` (`web/composer/prompts.py:207`)
    already serializes `state` and takes cross-session context as explicit kwargs
    (that is exactly how `schemas_loaded` — an in-memory, per-process,
    per-session signal — is threaded). The debrief rides the **same channel**:
    thread it as one new `build_context_string` kwarg, sourced from the reset
    seed-state's persisted `composer_meta`. This channel is inherently
    LLM-visible and user-hidden (it is the system prompt, not a chat message).
    Note: `CompositionState.to_dict()` does **not** carry `composer_meta`, so this
    is a *small plumbing addition* (one new context input), not a free ride — but
    still **no new role, no chat row, no frontend change**. Same family as Stage 1:
    shaping what the model sees, not adding a transcript surface.
  - *Durability / audit* → a **separate audit record** (below), not a chat row.
- **C1 + audit home — RESOLVED approach (one operator confirm): one reset event,
  double duty.** `archive_session` (`service.py:2206-2232`) hard-deletes sessions
  with no durable history (no `runs` / `composer_completion_events` row) and only
  soft-archives those that have one. Write a **single `session_reset` event to
  `composer_completion_events`** *before* calling `archive_session`: it (a) makes
  the prior session take the soft-archive branch (never hard-deleted) — satisfying
  C1 — and (b) **is** the durable audit record of the reset + debrief — satisfying
  the audit half of C2. The only ceremony is **one closed-enum extension** to
  `ck_composer_completion_events_type` (add `session_reset`, which — like the
  existing `export_yaml` — carries NULL `payload_digest`/`expires_at`, so the
  existing biconditional CHECKs are satisfied without change). Recommend this over
  a new table. **Remaining operator call:** approve the `event_type` extension.
  > **RESOLVED (2026-05-31 plan review): operator APPROVED the `session_reset`
  > `event_type` extension.** Scope of approval is narrow — only the
  > `session_reset` value carrying NULL `payload_digest`/`expires_at` (like
  > `export_yaml`). Two implementation corrections from the same review: (1) the
  > Python mirror of the verb set is `_CompletionVerb` `Literal` +
  > `_KNOWN_COMPLETION_VERBS` at `web/composer/telemetry_phase8.py:145-146`, **not**
  > a `StrEnum` in `contracts/`; (2) there is **no** `record_completion_event`
  > service method — the row is written by a direct insert into
  > `composer_completion_events_table` (see the `export_yaml` site at
  > `routes/composer.py:1218`). Adding the column also requires a
  > `SESSION_SCHEMA_EPOCH` bump (`web/sessions/schema.py`), and the resulting
  > `data/sessions.db` delete-and-recreate is a **gated destructive OPERATOR
  > ACTION**. See the Stage 3 plan's "Operator gates & pre-execution corrections".
- **C3 — no `reset_from` column** on `sessions`; needs a schema add (project DB
  policy: delete-and-recreate, no migration) + a new `SessionRecord` field, or
  overload `forked_from_session_id`.
- **C4 — `composition_states.provenance` is a closed enum** (`models.py:413`);
  `session_reset` is not a member. **Decision: reuse `session_seed`** for the
  empty seed-state unless the 8-step provenance-ceremony is justified.
- **C5 — enum value names differ from the spec draft:** user-uploaded blobs =
  `created_by == 'user'` (`models.py:1208`); accepted interpretation decisions =
  `choice IN ('accepted_as_drafted','amended')` (`models.py:680`, Python-side
  filter — `list_interpretation_events` has no "accepted" status). `copy_blobs_for_fork`
  copies *all* ready blobs at the **route** layer, so the user-uploaded filter is
  route-side. Debrief LLM synthesis: write a thin helper over `_litellm_acompletion`
  + `composer_advisor_model`; do **not** reuse `_call_advisor_with_audit`
  (`service.py:3104`, hardcoded stuck-prompt shape).

## Landing order

1. **Stage 1** (payload trim) — smallest, highest-leverage, no UI / DB change;
   directly kills the observed stall.
2. **Stage 2** (wire-visible gate + dedupe) — behavioural change to the compose
   loop; establishes the `blocked_pending_human` state.
3. **Stage 3** (reset + background debrief) — builds on Stage 2's taxonomy and
   the existing session/blob plumbing.

Each stage is independently shippable and independently testable.

## Test strategy

- **Stage 1:** the real failed session is the regression fixture. Assert: first
  `csv` option-shape rejection carries `plugin_schemas`; the *second* carries
  none but carries the explicit pull-directive (D2) and the verbatim error.
  Unit-test the delivery-aware gate with delivered/not-delivered sets.
- **Stage 2:** unit-test the declarative wire-visible enumeration per plugin
  (web_scrape + at least one sink that uses `reject_placeholder_value`). Test
  that choosing such a plugin with an unset/placeholder wire-visible field
  produces a `blocked_pending_human` entry and an up-front clarifying question,
  and does NOT produce a `rejected` entry (D3). Assert `implicit_decisions.py`
  classification still matches after the hardcoded list is removed (regression).
- **Stage 3:** test reset produces a new session with empty state, archived
  prior, carried-forward user blobs + accepted decisions, and a debrief seed
  that contains the deterministic facts, flags `blocked_pending_human` as
  not-`rejected`, marks synthesized portions, and contains no 5 KB schema (D1).
  Test that a terminal `blocked_pending_human` stop emits the debrief with no
  human-facing surface (tutorial-invisible).
- **Integration:** production code paths only (`ExecutionGraph.from_plugin_instances()`
  / `instantiate_plugins_from_config()` where the pipeline is exercised).

## Risks and mitigations

- **R1 — Suppressing the dump regresses convergence (the old problem).**
  Mitigated by D2 (explicit pull-directive) + a code comment citing the
  rationale + a regression test asserting the directive is present.
- **R2 — Dedup of `implicit_decisions.py` drifts audit classification.**
  Mitigated by a regression test pinning the classification output before/after
  the refactor.
- **R3 — Debrief synthesis leaks unflagged inference into audit.** Mitigated by
  D4 (`synthesized: true`) and a test asserting deterministic vs synthesized
  separation.
- **R4 — Reset orphans pending interpretation events.** Mitigated by the
  new-session boundary (not wipe-in-place) and a test for phantom-blocker
  absence on the new session.
- **R5 — CI gate fallout (tier-model allowlist, plugin source_file_hash,
  freeze guards).** Editing plugin files (Stage 2) requires refreshing
  `source_file_hash` and may rotate tier-model fingerprints; co-land per the
  project's gate-reconciliation procedure.

## Open items for the implementation plan

- Exact carrier field + injection point for the delivered-schema set (Stage 1).
- The precise declarative mechanism for wire-visible-required fields (classmethod
  vs Pydantic field metadata) and how `implicit_decisions.py` consumes it
  (Stage 2).
- The `reset_session` service method signature, route, and the
  `writer_principal` value for the debrief seed message (Stage 3).
- Whether the Landscape reset event is a new event type or rides an existing
  session-lifecycle event.
