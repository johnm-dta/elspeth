# Tutorial staged recut — design

- **Date:** 2026-06-22
- **Status:** Approved (brainstorm complete; revised after code-review adjudication) — ready for implementation planning
- **Branch:** `worktree-tutorial-staged-recut` — branched from `release/0.7.0`
  (merge-base `4bbb0624b`; `release/0.7.0` tip at design time `3074b2623`).
  `4bbb0624b` is the merge-base, **not** the branch tip.
- **Target version:** 0.7.0 (pre-release; no backward-compat / no feature flag)
- **Supersedes:** the single-pass "big-bang" first-run tutorial designed in
  `docs/composer/ux-redesign-2026-05/04-first-run-tutorial.md` (esp. line ~232,
  "the user describes the whole pipeline at once; multi-step is what guided mode
  is for") and `21-phase-4-hello-world-tutorial.md`.

### Revision history
- **rev 2 (2026-06-22):** Adjudicated a code review against the codebase. Material
  corrections: the canonical seed is `web_scrape` so recipe-match cannot fire today
  (we add a web_scrape recipe — D11); per-stage interpretation review reuses the
  freeform `interpretation_events` store, **not** a new guided turn type (D12); the
  advisor end sign-off must gate the terminal stamp *inside* the wire-commit branch
  (D13); the wire stage is a **global** engine change, not profile-only (D14); the
  v6 bump must also raise `SESSION_SCHEMA_EPOCH` for a loud boot guard (D15); plus
  wire-data-contract, cache-key, prompt-scoping, and entry-protocol corrections.
- **rev 3 (2026-06-22):** Adjudicated a second, deeper code review (14 findings).
  Material corrections: **web_scrape is a TRANSFORM, not a source** — the recipe
  predicate matches the URL-row `inline_blob` source, not `web_scrape` (D11 fixed);
  the interpretation-surfacing pass must fire after **every** site-creating commit
  (source, transform, **and the deterministic recipe-apply**) and must cover **all
  five kinds**, not just prompt_template — plus an explicit frontend **projection +
  blocking** rule in the guided branch (D12 expanded); the advisor end-gate is
  **fully fail-closed on every non-CLEAN verdict** (reversing rev-2's fail-open
  soft-proceed on UNAVAILABLE, which contradicted D7 — D13 corrected); the advisor
  pass counter must be **persisted** on `GuidedSession` or the re-entry bound never
  engages across HTTP requests (D16); the recipe must record an **audited
  prompt-injection-shield suppression** (D11); cache key **retains** the deployment
  hash and **adds** the `recipe_match` predicate hash; a **profile strip-on-fork**
  rule; the wire view needs **two reads** (topology + edge_contracts overlay); the
  profile must ride the **wire response**; §8 references the existing DB-reset
  runbook; plus test/gate and file-map corrections.

## 1. Motivation

The shipped first-run tutorial is a **big-bang**: the user types one canonical
sentence, and the composer LLM wires source + transforms + sink in a single
inference (`TutorialTurn2Describe` → `set_pipeline`, then auto-accepts all
proposals; a 3-assumption batch review in `TutorialTurn2bShowBuilt`). It has four
named failure modes (operator, 2026-06-22):

1. **Fragile below top-tier models.** One prompt crams source + sink + transform
   + wiring rules into a single inference; a model weaker than the frontier tier
   drops or mangles rules.
2. **Doesn't teach the mental model.** The user never learns that an elspeth
   pipeline *is* source → transform → sink → wiring.
3. **Magic box.** One prompt in, finished pipeline out, no visible logic.
4. **Slow → looks terrible.** One giant inference is high-latency and it is the
   product's first impression.

The recut is **two coupled decompositions**: a *rule/prompt* decomposition
(per-stage skill blocks, the lever on #1/#3) and a *flow/UX* decomposition (a
staged wizard that shows the logic, #2/#3). The latency lever (#4) is addressed by
(a) staged interaction-masking (each step's wait is hidden behind user
interaction, vs one long magic-box spinner), (b) a **new web_scrape recipe** that
lets the canonical pipeline compose deterministically with zero LLM calls (D11),
and (c) the existing run-cache that freezes the run-time rating.

## 2. Key discovery that shapes the design

The staged wizard already exists. `src/elspeth/web/composer/guided/` is a shipped,
audited staged source → sink → recipe-match → transforms wizard:

- **Per-stage rule blocks already carved out:** `guided/skills/base.md`
  (cross-cutting) + `step_1_source.md`, `step_2_sink.md`, `step_2_5_recipe_match.md`,
  `step_3_transforms.md`. `guided/prompts.py:load_step_chat_skill(step)` injects
  **base + exactly one stage's playbook** — used by the per-step **chat** solver.
  **Caveat (H1):** the transform **chain** solver (`chain_solver.py:172`) uses
  `load_guided_skill()` (`prompts.py:88`), which concatenates base + **all** step
  playbooks (whole-skill, by design — "the chain solver historically receives the
  full playbook for breadth"). So the most fragility-sensitive inference (the
  transform/wire solve) is **not** per-stage-scoped today, and adding
  `step_4_wire.md` to `_STEP_PLAYBOOK_ORDER` *grows* that prompt. We accept this
  growth (one stage block is a modest, deliberate increment); scoping the chain
  solver to relevant blocks is an optional future optimisation, not part of this
  recut. The per-step scoping is still the real lever for the source/sink/wire
  *chat* turns; the transform chain solve trades scoping for breadth on purpose.
- **Backend stage state machine:** `GuidedStep` (`guided/protocol.py:96-103`),
  `step_advance` (`guided/state_machine.py:508-554`), per-step widgets, a stepper.
- **A deterministic recipe-match short-circuit** (`step_2_5`) — but it matches
  **CSV sources only** (`composer/guided/recipe_match.py:186` `_is_csv`; all
  recipes hardcode `source.plugin == csv`, `composer/recipes.py:219/314/469`). The
  canonical pipeline's source is an **`inline_blob`** URL list feeding a
  **`web_scrape` transform** (web_scrape is a transform, not a source —
  `plugins/transforms/web_scrape.py:146`), so no current recipe fires for it; see
  §3/D11 (we add a web_scrape-shaped recipe keyed on the `inline_blob` source).
- **Interpretation review** is a **freeform** construct, **not** a guided turn type
  (corrected — see B4/D12). The backend `TurnType` enum (`protocol.py:16-25`) has
  six members and none is `interpretation_review`; that string exists only in the
  frontend union (`guided.ts:18`) as a dispatch case fed from the
  `interpretation_events` store, not from a backend-emitted guided turn. Today's
  tutorial already surfaces reviews this way (`TutorialTurn2bShowBuilt` +
  `interpretationEventsStore`). The recut reuses that mechanism (D12).

The advisor reads a wholly separate skill source (the monolithic
`composer/skills/pipeline_composer.md` via `build_system_prompt`), so it is
already decoupled from the per-stage blocks.

## 3. Decision

**Co-restructure both (Approach A — generalise guided in place):** the existing
`composer/guided/` state machine becomes *the* staged workflow engine,
parameterised by a `WorkflowProfile`. **The tutorial is a specific instance of
that engine** (profile = canonical seed + coaching + recipe-match + bookends);
live guided mode is the engine with the **empty** (canonical default) profile.

We **do not** relocate guided into a neutral `staged/` package now (Approach C —
deferred; the profile boundary makes it a later mechanical move). We **do not**
build a profile registry, a profile picker, or any "add workflow" UI — only the
plumbing (the profile parameter + one wired instance). A future profile is
*write-a-profile-object + wire its entry point*, touching no engine code. That
additivity is the acceptance test for the profile boundary.

### Decisions log

| # | Decision | Choice |
|---|----------|--------|
| D1 | Tutorial ↔ guided relationship | **Co-restructure both** — one engine, tutorial = instance |
| D2 | Refactor depth | **Approach A** — generalise guided in place (C deferred) |
| D3 | Stage order | **source → sink → transform** (both-ends-pinned field-reconciliation rationale) |
| D4 | Wiring | Its own **rule block** AND its own **user-facing stage** (4th stage: `wire`) |
| D5 | Advisor pack | Stays **big-bang monolith** in both end-gate and per-phase escape |
| D6 | Advisor firing | **Auto whole-pipeline END sign-off** + **per-phase on-demand "go to advisor"** escape |
| D7 | Staged end-gate polarity | **Blocking** (match freeform fail-closed); re-emit a revise turn on flag |
| D8 | Profile scope | **Plumbing only** — no registry/picker/menu; one wired instance |
| D9 | Frontend | Tutorial **drives the real coached guided stepper** |
| D10 | Migration | **In-place** for 0.7.0; remove big-bang components; no feature flag |
| **D11** | Canonical compose path | **Add a web_scrape recipe** (+ predicate) so the canonical pipeline composes deterministically (zero-LLM). **web_scrape is a TRANSFORM** — the predicate matches the URL-row `inline_blob` SOURCE; the builder emits source→web_scrape→llm_rate→field_mapper(cleanup)→jsonl. The recipe records an **audited prompt-injection-shield suppression** (no shield node; elspeth-abb2cb0931). Generalises recipe-match to web-scrape guided users |
| **D12** | Per-stage review surfacing | **Reuse the freeform `interpretation_events` store** — NO new backend guided `TurnType`. Surface after **every** site-creating commit (source, transform, AND recipe-apply), covering **all five kinds** (prompt_template, model_choice, invented_source, vague_term, pipeline_decision). Frontend: the guided ChatPanel branch must **project** pending events from `interpretationEventsStore` and **block advancement** while pending (the `GuidedTurn.tsx` `interpretation_review` case is dead code, not the path) |
| **D13** | Advisor terminal ordering + verdicts | END sign-off is a **pre-terminal gate inside the wire-commit branch**. CLEAN stamps COMPLETED. **Every non-CLEAN verdict** (FLAGGED, MALFORMED, UNAVAILABLE, timeout) is **fail-closed**: re-emit the wire turn while passes remain, then fail-closed non-runnable on budget-exhaustion. **No manual-proceed bypass** (reverses rev-2's fail-open). `STEP_4_WIRE` re-enterable, bounded by a **persisted** pass counter (D16) |
| **D14** | Stage set ownership | The stage SET is a **global engine property** (the frozen-total `GuidedStep` enum), NOT a `WorkflowProfile` field; live guided also gains the wire stage (intended improvement, not a parity regression) |
| **D15** | Schema migration mechanism | **Purge** (not migrate) AND bump `SESSION_SCHEMA_EPOCH` 22→23 so boot fail-closes loudly. Session-DB-only cutover; reuse the existing `docs/runbooks/staging-session-db-recreation.md` |
| **D16** | Duplicate-submission protection | **Minimal + concurrency parity.** Rely on existing per-session `compose_lock` + `step_advance` + terminal-409 + run-cache; **persist `advisor_checkpoint_passes_used`** on `GuidedSession` (same v6 bump) so duplicate advisor confirms can't re-call the provider unbounded; AND give `/guided/respond` the optimistic-concurrency `step_index` 409 guard `/guided/chat` already has. NO full idempotency-key infra (YAGNI) |

## 4. Architecture

```
        ┌─────────────────────────────────────────────┐
        │   Staged workflow engine (today's            │
        │   composer/guided/ state machine)            │
        │   reads a  WorkflowProfile  — hardcodes      │
        │   nothing tutorial-specific (except the      │
        │   stage SET, which is a global engine prop)  │
        └─────────────────────────────────────────────┘
                 ▲                          ▲
        EMPTY profile               TutorialProfile
        = live guided               = instance #1
        (today's behaviour           (canonical seed, coaching,
         MODULO the new wire stage)   recipe-match, bookends)
```

`WorkflowProfile` is internal plumbing. The engine consumes it; the tutorial
constructs the one concrete profile at its entry point; the empty profile is a
**canonical value, not `None`**. **Parity claim (corrected, D14):** the empty
profile reproduces today's live-guided behaviour **modulo the new wire stage** —
the stage SET is global, so live guided also gains `STEP_4_WIRE` (a benign,
intended improvement). The profile does **not** carry a `stages` field.

### 4.1 Stage flow

`GuidedStep` gains a fifth member **appended after transforms** (append is safe —
it only adds an ordinal; a mid-insert would renumber the wire protocol and is
forbidden):

| # | Stage | Elicits | Rule block |
|---|-------|---------|------------|
| 1 | Source | "where the data comes from" | `step_1_source.md` (exists) |
| 2 | Sink | "where it's going" | `step_2_sink.md` (exists) |
| 2.5 | *recipe-match* | deterministic short-circuit; **CSV today, + web_scrape via D11** | `step_2_5_recipe_match.md` (exists) |
| 3 | Transform | "what happens to it" | `step_3_transforms.md` (exists) |
| 4 | **Wire** *(new)* | "see how the pieces connect" + confirm | `step_4_wire.md` (**new**) |

The tutorial profile wraps this with **welcome** and **graduation** bookends
(frontend); the existing **run → audit** tail is unchanged. With the new
web_scrape recipe (D11), the canonical pipeline reaches the wire stage via the
deterministic recipe-apply path (zero LLM calls at compose time); a non-canonical
web_scrape pipeline still falls through to the live chain solver.

### 4.2 Adding the wire stage — coordinated touchpoints (verified)

Must hit all of these in lockstep (import-time asserts catch omissions):

- `guided/protocol.py:96-103` — add `STEP_4_WIRE` to the `GuidedStep` StrEnum.
- `guided/protocol.py:169-192` — add the step to `_LEGAL_TURN_MATRIX`; add the new
  `TurnType`(s) (`CONFIRM_WIRING`, and a revise turn per D13 — e.g. reuse
  `CONFIRM_WIRING` with an attached `advisor_findings` payload) to the enum
  (`:16-25`) plus `_REQUIRED_KEYS` (`:200-218`) and `_NESTED_SHAPES` (`:243-253`),
  both **total** over `TurnType` (omission crashes at import).
- `guided/prompts.py:43-58` — add the step to `_STEP_FILE_NAMES` and
  `_STEP_PLAYBOOK_ORDER` and create `guided/skills/step_4_wire.md`, or the
  import-time asserts (`:64-73`) crash. (Note H1: this grows the whole-skill chain
  solver prompt — accepted.)
- `guided/emitters.py:428-432` **and** `sessions/routes/_helpers.py:3674-3678` —
  append the step to **both** duplicated `_ORDER` ordinal tuples; add a
  `build_step_4_wire_turn` emitter.
- `guided/state_machine.py:508-554` — add the `STEP_4_WIRE` branch + `_advance_step_4`
  (incl. its **self-loop** for re-entry across advisor sign-off rounds, D13);
  change `_advance_step_3`'s accept path to advance to `STEP_4` instead of
  terminating.
- `sessions/routes/_helpers.py:3339` (route `STEP_3` block) — redirect chain-accept
  away from immediate `COMPLETED` to emit the wire turn; add the `STEP_4_WIRE`
  dispatch branch that runs the advisor gate (D13) and only then stamps `COMPLETED`.
- **Both** completion seams must redirect: `handle_step_3_chain_accept`
  (`steps.py:390-406`) **and** `handle_step_2_5_recipe_apply` (`steps.py:262-274`)
  stamp `COMPLETED` today; **move the terminal-stamping out of both** and into the
  `STEP_4_WIRE` handler, or the recipe path skips wiring.
- `sessions/routes/composer.py:1366-1414` — add the `STEP_4_WIRE` rebuild branch in
  the GET `/guided` re-fetch dispatcher.
- `frontend/src/types/guided.ts:18-39` — add the step string to the hand-written
  `GuidedStep` TS union and the new `TurnType`(s) to the `TurnType` union.

### 4.3 WorkflowProfile threading + entry protocol (greenfield)

- Add a frozen, persisted `profile: WorkflowProfile` field to `GuidedSession`
  (`state_machine.py:257-332`): bump `GUIDED_SESSION_SCHEMA_VERSION` **5 → 6**, add
  the field with a default = the empty/canonical profile, add strict
  `to_dict`/`from_dict` lines (direct-key, no `.get()`). See §8 for the migration.
- Parameterise `GuidedSession.initial(profile)` (`state_machine.py:342-351`) and
  thread the profile into `_initial_composition_state_with_guided_session`
  (`_helpers.py:2183-2208`).
- **Entry protocol (B2 — the missing link).** Today guided entry is no-arg/lazy:
  `GET /{session_id}/guided` (`composer.py:1455-1508`) takes no profile input and
  lazily builds a default guided state; `createSession()` (`client.ts:329`) posts
  only `{title}`. Threading the parameter into the helper is necessary but not
  sufficient — nothing upstream supplies it. **Add a tutorial-start endpoint**
  (e.g. `POST /api/sessions/{id}/guided/start` with an optional closed-enum
  `profile` discriminator, or fold into the `/api/tutorial/*` route family) that
  constructs the session via `_initial_composition_state_with_guided_session(
  profile=TUTORIAL_PROFILE)` and **persists** it. `GET /guided` then **reads** the
  persisted `GuidedSession.profile` (round-tripped by the v6 schema) rather than
  re-constructing a default. The frontend tutorial machine calls this start
  endpoint instead of relying on the lazy GET default. The lazy no-arg GET path
  stays untouched for live guided (empty profile).
- Gate recipe-match toggle (STEP_2_5 dispatch, `_helpers.py:3053-3161`), advisor
  checkpoints (`solve_chain` call sites `_helpers.py:3112/3352`), and welcome/
  graduation copy on profile flags. **The stage SET is NOT gated on the profile**
  (D14) — it is a global engine property governed by the frozen-total enum.

`WorkflowProfile` fields (initial): `entry_seed` (canonical prompt + pre-filled
source vs empty), `coaching` (per-stage copy, on/off), `advisor_checkpoints`
(on/off), `recipe_match` (on/off), `bookends` (welcome/graduation). **No `stages`
field** (D14). The default (empty) profile sets all to live-guided behaviour.

- **Surface the profile on the wire (finding 5).** The chrome can't render coaching/
  bookends if the profile only lives backend-side. Add `profile: WorkflowProfileResponse
  | None` to the shared `GuidedSessionResponse` (`schemas.py:323`; `None` for the
  empty/live-guided profile) — it threads through `GetGuidedResponse` /
  `GuidedRespondResponse` / `GuidedChatResponse` automatically (all embed
  `guided_session`). Mirror it on the TS `GuidedSession` (`guided.ts`) as
  `profile: WorkflowProfile | null`. Wire-visible subset = `coaching` + `bookends`
  (+ a `recipe_match`/`advisor_checkpoints` hint); `entry_seed` is consumed
  server-side at start and need not ride GET.
- **Persist the advisor pass counter (D16).** Add `advisor_checkpoint_passes_used:
  int = 0` to `GuidedSession` in the **same** v6 bump (strict `to_dict`/`from_dict`).
  Today it's a per-compose local (`service.py:3160`); guided re-entry is across
  separate `/guided/respond` requests, so an unpersisted counter resets to 0 each
  request and `composer_advisor_checkpoint_max_passes` **never bounds re-entry**. The
  `STEP_4_WIRE` branch reads/increments/persists it (this is what makes D13's "bounded
  re-entry" real).
- **Profile lifecycle — strip on fork (finding 10).** `composer_meta['guided_session']`
  (now carrying `profile`) is copied **verbatim** on fork (`service.py:5076`; blob
  re-save `sessions.py:489`, "fork inherits the operational provenance") and
  round-tripped on resume — so a persisted `TutorialProfile` would **leak** into a
  forked/resumed ordinary guided session. Rule: **reset `GuidedSession.profile` to the
  empty/canonical profile at the `composer_meta` copy seam on fork** (single chokepoint
  covering both copy paths) unless the fork is an explicit tutorial continuation.
- **Duplicate-submission / concurrency (D16).** Inherited guarantees (state them so
  the plan doesn't reinvent them): all guided mutations are serialized per-session by
  `compose_lock` (`_helpers.py:212`); duplicate `/guided/respond` is naturally
  idempotent because `step_advance` moves `guided.step` and the terminal **409 guard**
  (`composer.py:2131`) rejects post-terminal re-submits; `tutorial/run` dedups via the
  run-cache (no provider re-call). Add: give `/guided/respond` the **optimistic-
  concurrency `step_index` 409 guard** that `/guided/chat` already has
  (`composer.py:2658`) — carry an expected step on the wire confirm, 409 on mismatch.
  No idempotency-key infrastructure (YAGNI for a single-user-per-session wizard).

## 5. Backend mechanics

### B1 — Per-stage interpretation review (fixes a latent silent-orphan bug)

**Latent bug (verified):** the freeform fail-closed orphan gate
(`service.py:_missing_pending_interpretation_review_sites`, `:1376`) runs **only**
inside the freeform no-tool finalize loop (`_try_terminate_no_tools`, `:2565`).
Guided commits each step by calling `_execute_*` tools directly (`steps.py`) and
**never enters that loop**. So a guided step 3 that commits an `llm` node creates
real interpretation sites that are **surfaced to no one** and only fail at *run*
time with `UnresolvedInterpretationPlaceholderError` (`execution/service.py:514-525`)
and zero pending events — the staging-500 class. The recut hits this the moment a
stage adds an LLM transform, so the fix is in-scope.

**Fix / design (corrected, D12).** Add an explicit **surfacing pass** in the guided
dispatcher that fires **after every guided mutation that can create interpretation
sites** — not just transform-commit. Concretely: after the **source** commit
(`invented_source` for an LLM-authored source — `interpretation_state.py:524-555`
proves a source CAN produce a site, so the rev-2 "source-only yields zero sites" was
wrong; it's only zero for the *non-LLM* `inline_blob` URL source), after the
**transform** commit (`handle_step_3_chain_accept`), and after the **deterministic
recipe-apply** commit (`handle_step_2_5_recipe_apply` — which auto-stages
`llm_prompt_template` + `llm_model_choice` via `_options_with_default_llm_reviews`
then stamps `COMPLETED` today, a silent orphan). The pass enumerates
`interpretation_sites(state)` on the post-mutation state and surfaces the delta.

**Recipe-apply must redirect through the wire stage** (per §4.2), not stamp
`COMPLETED` directly — otherwise the deterministic path skips both surfacing and the
wire/advisor gates.

**Kind coverage — all five kinds, not just prompt_template.**
`_auto_surface_prompt_template_reviews` (`service.py:1412`) handles **only**
`llm_prompt_template`; the other four kinds (`invented_source`, `vague_term`,
`llm_model_choice`, `pipeline_decision`) surface today *only* when the LLM emits
`request_interpretation_review` (`sessions.py:1659`) — which **never fires on the
zero-LLM recipe path**, so they orphan and block the run
(`_missing_model_choice_review_sites:675` proves model_choice sites block). Extend
the surfacer into a **kind-general** backend surfacer (or sibling surfacers) covering
all five kinds for the LLM-less path. All write `interpretation_events` rows surfaced
via the existing store/UI (D12) — **no new guided `TurnType`.**

**Frontend projection + blocking (D12, F4).** The `GuidedTurn.tsx`
`interpretation_review` case is **dead code** — the guided ChatPanel branch
(`ChatPanel.tsx:1109`) renders only `guidedNextTurn`, so it is *not* the projection
path. Mirror what `TutorialTurn2bShowBuilt` already does: the guided branch must read
`interpretationEventsStore.pendingBySession[sessionId]`, filter to pending
`user_approved` events, render each via `InterpretationReviewTurn`, and **block
advancement** (gate the wire-stage confirm + the Continue button, the disabled-pattern
at `TutorialTurn2bShowBuilt.tsx:128`) while any pending remains.

Polarity: per-stage review is **surface-and-resolve (advisory)** at commit; the
existing run-time gate (`UnresolvedInterpretationPlaceholderError`,
`execution/service.py:514-525`) remains the hard backstop, so we do not re-implement
fail-closed at step commit. Watch: the raw-HTML cleanup contract
(`composition_review_contract_error` inside `_execute_set_pipeline`,
`sessions.py:657`) is already BLOCKING for `web_scrape → field_mapper` raw-field
drops — the canonical recipe (D11) must stage the `pipeline_decision` or the commit
hard-fails (400).

### B2 — The wire stage's data model (corrected)

No new backend/endpoint needed, but the wire view needs **two reads** (finding 11) —
**neither tool alone suffices**:
1. **`get_pipeline_state`** for the **connection-label topology**:
   `_serialize_full_pipeline_state` (`sessions.py:1071-1085`) carries each node's
   `input / on_success / on_error / routes / fork_to` and each source's `on_success`
   (`_common.py:968-1010`). `preview_pipeline`'s own `nodes` list is only
   `{id, node_type, plugin}` — **not** a topology source.
2. **`preview_pipeline`** for the **`edge_contracts` (+ `semantic_contracts`)
   overlay**: built by `_authoring_validation_payload` (`sessions.py:1153-1162`),
   surfaced only by `_execute_preview_pipeline`
   (`composer/tools/generation.py:1651`; summary at `:1685-1692`).
   `get_pipeline_state` runs no `validate()`, so it has no `edge_contracts`.

**Client path (preferred):** the guided dispatcher returns **both** blobs on the
`STEP_4_WIRE` turn payload (one round-trip, no new client wiring) rather than the
frontend issuing two reads. Add a TS `WireStageData` type to `guided.ts`:
`{ topology: { sources: Record<string,{plugin; on_success}>; nodes: Array<{id;
node_type; plugin; input; on_success; routes; fork_to}>; outputs: Array<{sink_name;
plugin}> }; edge_contracts: Array<{from; to; producer_guarantees; consumer_requires;
missing_fields; satisfied}>; semantic_contracts: ... }`. Render: reconstruct edges
from connection labels (`source.on_success → node.input`; `node.on_success/routes/
fork_to → downstream`) and overlay `edge_contracts` keyed by `(from, to)` to colour
satisfied/unsatisfied + show `missing_fields`.

**Serialized wire keys (M1 correction):** `EdgeContract.to_dict()`
(`state.py:359-368`) emits keys **`from` / `to`** (the dataclass *fields* are
`from_id`/`to_id`, but `to_dict` renames them). The wire view must read
`{from, to, producer_guarantees, consumer_requires, missing_fields, satisfied}` —
**not** `from_id`/`to_id`.

**Hard constraint (the one trap):** wiring is carried by **named connection
labels** (`source.on_success` / `node.input` / `node.on_success` / `routes` /
`fork_to`), **not** `EdgeSpec` objects (guided passes `edges=[]`, `steps.py:359`).
The wire view **reconstructs topology from connection labels and overlays
`edge_contracts`** — never renders `state.edges` directly.

**"Confirm wiring" does not commit routing** (the step handlers already wired it).
It (1) gates progression to `COMPLETED`/sign-off on `validate().is_valid` (zero
blocking field-contract errors, all `edge_contracts.satisfied`), and (2) lets the
user accept a reconciliation — insert a `field_mapper` or relax a schema to
`flexible` — via existing `upsert_node` / `set_pipeline` tools. **After any such
reconciliation the confirm gate re-evaluates `validate().is_valid` AND re-runs the
B1 per-stage surfacing pass on the post-mutation state** (B6 — never trust
transform-commit-time results at the wire terminal); the run-time interpretation
gate re-derives from the final state regardless, so an advisory card that goes
cosmetically stale across a wire mutation cannot cause a bad run. Honest-gap
rendering: coalesce/fork nodes skip `edge_contracts` ("not statically checkable");
shape-changer produced-fields may be unavailable mid-edit. The canonical pipeline
has none of these.

### B3 — Advisor wiring (end sign-off + per-phase escape; corrected ordering)

The advisor's `_run_advisor_checkpoint(phase='end')` (`service.py:4176`) +
`_build_checkpoint_arguments(phase='end')` (`service.py:4108`) is state-driven,
non-raising, and already loads the **big-bang monolith** via
`build_system_prompt(self._data_dir)` (`service.py:3954`) — decoupled from the
per-stage skills. It is a **private `ComposerServiceImpl` method absent from the
`ComposerService` Protocol** (`protocol.py:703`), and the staged dispatcher holds
no service handle — so add a **public protocol method** (e.g.
`async def run_signoff_checkpoint(state, session_id, recorder, progress) ->
AdvisorCheckpointVerdict`) delegating to the private one, and thread the handle
into the route/dispatcher.

**Terminal ordering (B5/D13 — corrected).** The END sign-off must be a
**pre-terminal gate inside the `STEP_4_WIRE` dispatch branch**, *before* it stamps
`COMPLETED` — **not** a post-`terminal = guided.terminal` hook at
`composer.py:2381` (once terminal is stamped, `/guided/respond` rejects further
responses with 409 at `composer.py:2131`, foreclosing the revise turn). Behaviour:

**Verdict matrix (corrected, D13 — every non-CLEAN is fail-closed).** rev-2's
"soft re-emit / manual proceed on UNAVAILABLE" was a **fail-open** that contradicted
D7 (operator chose blocking) and the implemented binding-authority doctrine
(`service.py:2747` fails closed on `not verdict.ok`; comment at `:2711-2712`: "an
unavailable advisor … FAILS CLOSED — the mandatory final authority D-5/D-7/D-8").
Corrected — there is **no learner-driven manual-proceed**:

| Verdict | Terminal | Wire re-emit (while passes remain) | Affordance |
|---|---|---|---|
| CLEAN | COMPLETED | — | — |
| FLAGGED | None | yes | `findings_text`; "resolve & re-confirm" |
| MALFORMED / unparseable | None | yes | folds into not-CLEAN→flagged at `service.py:4217`; same as FLAGGED |
| UNAVAILABLE / timeout (after bounded retry) | None | yes | "advisor sign-off could not be obtained; retry" — **never** a manual bypass |
| Budget-exhausted (passes used up, still non-CLEAN) | **fail-closed non-runnable** (reuse `_advisor_signoff_blocked_validation`) | no | "resolve & re-run the composer" |

Timeout is an UNAVAILABLE sub-case (caught + retried inside `_run_advisor_checkpoint`,
bounded by `composer_advisor_timeout_seconds`). The non-dead-end concern that
motivated rev-2's fail-open is already satisfied **without** fail-open: because
`STEP_4_WIRE` is re-enterable, a transient blip just re-emits a retry turn — the
learner is never bricked *and* never bypasses authority. The run-time interpretation
gate is a backstop, not a substitute for sign-off.

`STEP_4_WIRE` is **re-enterable** (self-loop, bounded by the **persisted**
`advisor_checkpoint_passes_used`, D16 — the per-compose local would reset to 0 each
HTTP request and never bound re-entry). **Per-phase on-demand "go to advisor":**
ride the existing `ControlSignal.REQUEST_ADVISOR` (today a no-op re-prompt at step
3 only, `_helpers.py:3342`) — repoint it to the whole-pipeline checkpoint with a
"structurally complete enough to review" guard, extended to all stages. Keep the
trust tier correct: operator-triggered escapes must **not** route through the
Tier-3 `_validate_advisor_arguments`, and backend checkpoints must not consume
unvalidated user text.

**Budget/latency:** size `composer_advisor_checkpoint_max_passes` (checkpoint
budget), `composer_advisor_max_calls_per_compose` (escape-hatch budget),
`composer_advisor_timeout_seconds` (default 60s — the dominant added latency), and
`composer_timeout_seconds`. With D11, the canonical pipeline composes zero-LLM
(recipe-apply), so its only frontier round-trip is the single end sign-off; decide
whether that call is awaited inline vs streamed via progress, and revisit the 300s
staging ceiling. Also correct stale prose: `tools/_dispatch.py:129` says the
advisor is "Disabled by default" — contradicts the mandatory-advisor reality.

### B4 (D11) — The web_scrape recipe

**Correct the model: web_scrape is a TRANSFORM, not a source**
(`plugins/transforms/web_scrape.py:146` `class WebScrapeConfig(TransformDataConfig)`,
`:149` `url_field` = "the row field whose value is the absolute URL to fetch"). The
canonical pipeline is: an **`inline_blob` source** carrying rows of `{url: ...}`
(the five gov pages) → **`web_scrape` transform** → **`llm_rate` transform** →
**`field_mapper`** (`select_only: true`, drops the raw content/fingerprint fields)
→ **jsonl** sink. (Validated artifact: `tutorial.spec.ts:54-90`; the parallel real
example `examples/chaosweb/settings.yaml` uses a `csv` source with a url column.)

Today every recipe gates on `source.plugin == csv` (`recipe_match.py:186`
`_is_csv`; `composer/recipes.py:219/314/469`), so `match_recipe` returns `None` and
the dispatcher falls through to the live chain solver (`_helpers.py:3107`,
`solve_chain_with_auto_drop`). Add:

1. A predicate in `composer/guided/recipe_match.py` matching the **URL-row SOURCE
   plugin** (`inline_blob`) + single jsonl output + the url-column/required-fields
   signal — it must **not** reference `web_scrape` as a source. (Resolve the exact
   registered source-plugin name when implementing — `inline_blob` materialises to
   a concrete row source — and pin it with a test against the canonical composed
   state.)
2. A `RecipeSpec` in `composer/recipes.py` whose `_build_*` deterministically emits
   the full chain above, **naming the head source node**, and **stages the
   `kind="pipeline_decision"` raw-HTML cleanup requirement on the `field_mapper`
   node** (skill `pipeline_composer.md:958-1014,1233`) so it passes the B1 blocking
   cleanup contract (`sessions.py:657`).

**Prompt-injection-shield decision (audited suppression).** The canonical
`web_scrape → llm_rate` flow routes untrusted public-web text into an LLM. Because
the recipe is deterministic and zero-LLM, the recipe **author** makes the shield
decision up front (no runtime LLM judgement). On the record: the prompt-injection
shield control is **deliberately suppressed** under `elspeth-abb2cb0931` (the
composer cannot gate `azure_prompt_shield` on configured-secret availability, so
recommending it is misleading). Therefore the recipe ships **without** a shield
node and **without** a `prompt_injection_shield_recommendation` decision — this is
an **audited suppression, not an oversight**. Add a test asserting the built
pipeline contains no `azure_prompt_shield` node and stages no shield recommendation,
with a comment referencing `elspeth-abb2cb0931`, so the suppression is pinned and
cannot silently regress (when that ticket lands, the recipe gains an authorized
shield + decision). This generalises recipe-match to web-scrape guided users (a
co-restructure win) and is what makes the §4.1 "zero-LLM canonical compose" true.

## 6. Run / audit / cache tail

### C1 — Unchanged
`run_tutorial_pipeline` (`tutorial_service.py:82`, `POST /api/tutorial/run`) is
purely post-compose: it runs the persisted latest `CompositionStateRecord`,
agnostic to how it was composed. **Run → audit → graduation survive as-is.**

### C2 — Cache key inputs (corrected, H2)
`tutorial_model_id` (`tutorial_service.py:784-821`) currently folds
`composer_model` + the **freeform** `pipeline_composer.md` hash + deployment hash.
For the staged design it must fold the complete set of **operator-controlled
deterministic inputs**:

**Four** keyed inputs (finding 9 — rev-2 listed three and its "instead of the
freeform skill" phrasing silently **dropped** the deployment hash that the current
code keys and the staged advisor still consumes):

1. `composer_model`.
2. The **staged skill hashes** (`base.md` + `step_1..step_4_wire.md`) — folded **in
   addition to**, not instead of —
3. the **deployment-overlay hash** `sha256(load_deployment_skill("pipeline_composer",
   data_dir))`, **RETAINED** from the current `tutorial_model_id`: the staged advisor
   END sign-off still consumes `build_system_prompt(data_dir)` (`prompts.py:62-86`)
   and per D13 that verdict shapes the cached audit artifact.
4. A content hash covering **both** `composer/recipes.py` (`RecipeSpec` + `_build_*`)
   **and** `composer/guided/recipe_match.py` (the predicate registry + slot
   resolvers) — `recipe_match` selects which recipe fires and pre-fills slots, and
   under D11 the recipe **deterministically authors the cached pipeline including
   option-level content** (provider, model, prompt_template, response_field, schema
   mode, output format). The topology-match guard (`_state_matches_cached_topology`,
   `:713-732`) is **option-blind by design**, so it cannot catch recipe/predicate
   drift — keep it option-blind and guarantee option fidelity by **keying** every
   deterministic input instead.

Keep the topology guard as-is. Keep posting the canonical seed token at Turn-4 as
the cache key (the profile carries it). **Single-profile assertion:** exactly one
profile posts the canonical seed; any second profile MUST carry a distinct seed
token (add a test) so `canonical_prompt` stays the profile discriminator by
construction — folding full profile identity into `model_id` is deferred with the
second profile (D8).

## 7. Frontend (D9 — coached real stepper)

- **Survives:** `welcome` bookend and the `run → audit → graduation` tail.
- **Replaced:** `describe` + `showBuilt` (big-bang compose + 3-assumption batch
  review) → the staged guided walk (source / sink / transform / wire), each with
  its own per-stage interpretation review **via the existing `interpretationEventsStore`
  / `InterpretationReviewTurn`** (D12 — no new turn type).
- **Folds in:** the old `graph` turn is subsumed by the **wire** stage.
- **Collapses:** Turn-6 `mode` choice becomes a graduation affordance ("keep
  building in guided / try freeform").
- **Mechanism / bridge component (finding 5).** Add a named **`TutorialGuidedShell`**
  component that (a) renders the `welcome` bookend, (b) on Start calls
  `POST /guided/start` with the `TutorialProfile` and switches the session into guided
  mode, (c) renders the **same** guided surface as `ChatPanel`'s `chat-panel--guided`
  branch, and (d) on guided `terminal=completed` hands off to the surviving
  `tutorialMachine.ts` run/audit/graduation tail. **Recommended: EMBED** — the
  tutorial mounts the real `ChatPanel` guided branch (lowest new surface, maximal
  reuse, truest "use the real thing"); the alternative (EXTRACT a shared
  `<GuidedComposer>`) is only worth it if other callers need the shell. The chrome
  reads `coaching` / `bookends` off the **wire** `GuidedSession.profile` (§4.3).
- **Interpretation projection + blocking (D12, F4).** The guided surface must read
  `interpretationEventsStore.pendingBySession`, render pending `user_approved` events
  via `InterpretationReviewTurn`, and **block** the wire confirm + Continue while any
  pending remains (the `TutorialTurn2bShowBuilt.tsx:128` disabled-pattern). The
  `GuidedTurn.tsx` `interpretation_review` case is dead — not the projection path.
- **Accepted limitation:** the `welcome`/`run`/`audit`/`graduation` bookend progress
  stays **local** to `tutorialMachine.ts` (not persisted), so a hard refresh mid-tail
  restarts the tail while the guided step itself resumes from the persisted profile via
  `GET /guided`. Remove `TutorialTurn2Describe` / `TutorialTurn2bShowBuilt`;
  `tutorialMachine.ts` retains `welcome` + the run/audit/graduation tail.

## 8. Migration (D10, D15 — corrected runbook)

**Purge, do not migrate** (pre-release delete-stale-DB policy; operator holds a
standing grant to delete ELSPETH session DBs). The original "the v5→v6 bump already
forces a sessions-DB delete on deploy" claim was **wrong**: the `GuidedSession`
blob lives in the `composition_states.composer_meta` JSON column (no new SQL
column), so `GUIDED_SESSION_SCHEMA_VERSION` 5→6 is enforced **lazily** by
`GuidedSession.from_dict` (`state_machine.py:400-403`) as a per-row **HTTP 500** when
a stale guided session is opened — *not* a boot crash.

**Use the existing runbook — do not reinvent it.** `docs/runbooks/staging-session-db-recreation.md`
("Session DB Reset Runbook") is the canonical procedure: it already handles the
`-wal`/`-shm`/`-journal` sidecars as one artifact set, `fuser` open-handle checks, the
`RECREATE` operator-confirmation prompt, the `user_secrets` blast-radius sign-off, and
archive/rollback. This recut is a **session-DB-only** cutover (it bumps
`SESSION_SCHEMA_EPOCH`, **not** `SQLITE_SCHEMA_EPOCH` — the run/audit Landscape schema
is unchanged, §C1), so follow that runbook's **single-DB** path, not its Phase-5b
two-DB path. The epoch guard is **SQLite-specific** (`PRAGMA application_id` /
`user_version`); the staging deploy is single-worker SQLite, and Postgres is out of
scope/unproven for this cutover. Spec-level deltas on top of the runbook:

1. **Also bump `SESSION_SCHEMA_EPOCH` 22→23** (`models.py:117`) so
   `initialize_session_schema` → `_assert_schema_sentinels` (`schema.py:163-169`)
   **fail-closes at boot** with the actionable "Delete the session DB file and
   restart" message — converting the silent lazy-500 into a loud boot guard (D15).
2. **Exact target:** the sessions DB at `{data_dir}/sessions.db` — resolve the live
   path from `WebSettings`/`get_session_db_url()`, do **not** assume `./data`.
3. **Isolation:** `{data_dir}/runs/audit.db` and `{data_dir}/auth.db` are **separate
   files** and MUST NOT be deleted.
4. **WARNING — data-loss blast radius:** `UserSecretStore` rides the **session
   engine** (`app.py:874`), so per-user stored secrets live **in `sessions.db`** and
   are destroyed by this delete. Operators must re-enter per-user secrets after
   deploy (acceptable pre-release; must be stated).
5. **Backup:** archive the full SQLite artifact set — `sessions.db` + `-wal` + `-shm`
   + **`-journal`** — as one unit before deletion (the runbook's `cp -a` loop;
   reversible within the deploy window).
6. **Verification:** assert `PRAGMA user_version == SESSION_SCHEMA_EPOCH` on the new
   DB, plus a smoke test that creating + running a fresh guided/tutorial session
   reaches `COMPLETED` without a 500.

Keep the canonical "cool government pages" artifact so the validated end-to-end
pipeline still applies. No long-lived feature flag.

## 9. Testing

### 9.1 Test cases (TDD, load-bearing first)
- **Backend:**
  - **Interpretation surfacing — per kind, per boundary (D12):** at the **source**
    commit an LLM-authored source surfaces an `invented_source` card; at the
    **transform** commit a committed `llm` node surfaces `llm_prompt_template` +
    `llm_model_choice` cards; at the **recipe-apply** commit (zero-LLM path) the
    auto-staged kinds surface (not silent orphans). Each is a **resolvable card**, not
    a run-time 500. Cover all five kinds (incl. `pipeline_decision`, `vague_term`).
  - The new **web_scrape recipe (D11):** predicate matches the **`inline_blob`** URL
    source (not `web_scrape`); `_build_*` names the head source + emits
    source→web_scrape→llm_rate→field_mapper→jsonl, staging the raw-HTML
    `pipeline_decision` so it passes the blocking cleanup contract; recipe-apply
    redirects **through** the wire stage. **Shield suppression test:** the built
    pipeline has **no** `azure_prompt_shield` node and stages no shield recommendation
    (comment refs `elspeth-abb2cb0931`).
  - **Zero-LLM compose assertion:** the recipe-apply path makes **zero** LLM calls at
    compose time (assert `llm_call_count == 0` / no `_litellm_acompletion`) — the gate
    for the §4.1 zero-LLM claim.
  - Wire stage: topology reconstructed from connection labels (not `state.edges`);
    the two-read merge (`get_pipeline_state` + `preview_pipeline.edge_contracts`);
    `validate().is_valid` confirm gate; re-validate + re-surface after a
    `field_mapper`/schema-relax reconciliation (B6).
  - **Advisor END sign-off matrix (B3/D13 — all non-CLEAN fail-closed):** CLEAN →
    COMPLETED; FLAGGED/MALFORMED/UNAVAILABLE/timeout → re-emit wire turn while passes
    remain (terminal stays None); budget-exhausted → fail-closed non-runnable. Assert
    terminal **never** COMPLETED on a non-CLEAN verdict, **no** manual-proceed path,
    and pre-terminal ordering (no 409 dead-end).
  - **Duplicate-submission (D16):** a double-confirm at the wire stage results in
    **exactly one** advisor provider call (the persisted `advisor_checkpoint_passes_used`
    bounds re-entry across requests); a stale `step_index` on `/guided/respond` → 409.
  - **Per-phase `REQUEST_ADVISOR` escape:** dispatch from each stage; the trust-tier
    guard (operator-triggered escapes do NOT pass through Tier-3
    `_validate_advisor_arguments`; checkpoints do not consume unvalidated user text);
    a client-tamper case.
  - **Cache identity (C2 — four inputs):** editing a stage block, **the deployment
    overlay**, OR a `recipe_match` predicate/`recipes.py` builder each invalidates the
    cache; `_state_matches_cached_topology` tolerance; single-profile seed assertion.
  - **Profile lifecycle (finding 10):** fork a tutorial-profile session → the forked
    `GuidedSession.profile` is the **empty** profile (no tutorial seed/coaching/
    bookends); an ordinary guided fork is unaffected.
  - `WorkflowProfile` schema-v6 strict round-trip (incl. `advisor_checkpoint_passes_used`);
    a **pre-v6 GuidedSession is rejected** on load; the `SESSION_SCHEMA_EPOCH` 22→23
    boot guard fires.
- **Frontend:**
  - Extend `composer-guided.spec.ts` for the wire stage; rewrite `tutorial.spec.ts`
    to the staged flow (the 3-assumption batch gate becomes per-stage reviews via the
    interpretation store); rewrite `tutorial-reliability.staging.spec.ts`.
  - **E2E happy path:** recipe-apply/chain-accept → wire-confirm (`validate().is_valid`)
    → advisor sign-off CLEAN → `terminal=COMPLETED`, asserting no 409 dead-end and the
    COMPLETED stamp lands inside the wire branch.
  - **Interpretation lifecycle (E2E):** an **unresolved** per-stage card **blocks the
    run** (`UnresolvedInterpretationPlaceholderError`); resolving it via
    `interpretationEventsStore` **permits** the run to proceed — exercise both branches.
  - **Bridging (D9):** the `TutorialGuidedShell` mounts the **real** `ChatPanel`
    guided branch and reads `profile.coaching` / `profile.bookends` off the **wire**
    `GuidedSession` (not a tutorial-only stepper).
- **Parity guard (D14):** the empty profile differs from the tutorial profile ONLY
  in the profile-gated dimensions (entry_seed, coaching, advisor_checkpoints,
  recipe_match, bookends) — NOT byte-identical to the pre-recut 4-step flow (the enum
  now has 5 members for all profiles).

### 9.2 Gates / verification commands
- `uv run ruff check src/ tests/ scripts/ examples/ elspeth-lints/src/`
- `uv run ruff format --check src/ tests/ scripts/ examples/ elspeth-lints/src/`
- `uv run mypy src/ elspeth-lints/src/`
- **elspeth-lints trust gates (both rulesets this work trips):** `trust_tier.tier_model`
  (B1/B3 add/move Tier-3 advisor/user-text handling) **and**
  `trust_boundary.tests,trust_boundary.scope,trust_boundary.tier` (honesty-gate over
  new `@trust_boundary` decorations), plus the composer/plugin_contract/immutability/
  audit_evidence rules touched by wire/profile/recipe changes —
  `PYTHONPATH=elspeth-lints/src uv run python -m elspeth_lints.core.cli check --rules trust_tier.tier_model,trust_boundary.tests,trust_boundary.scope,trust_boundary.tier,'composer/*' --root src/elspeth`
- **`SlotType` / `guided.ts` mirror drift gate** (this work edits `guided.ts`):
  `.venv/bin/python scripts/cicd/check_slot_type_cross_language.py` (CI:
  "Check SlotType / guided.ts mirror consistency").
- targeted `pytest` over the new guided/tutorial/recipe test files
- **frontend (run from `src/elspeth/web/frontend`):** `npm run typecheck`,
  `npm test -- --run` (vitest), `npm run build`, Playwright `npm run test:e2e`
  (+ `test:e2e:staging`) — CI uses `working-directory: .../src/elspeth/web/frontend`.
- `wardline scan . --fail-on ERROR` (exit 0 clean / 1 gate tripped / 2 tool error) —
  B1/B3/recipe touch externally-fed advisor/user-text trust boundaries; this is the
  taint-flow gate, **separate** from the elspeth-lints trust-boundary honesty-gate
  above. See the `wardline-gate` skill at `.agents/skills/wardline-gate/SKILL.md`
  (not `docs/agents.md`; absent here).

## 10. Non-goals / out of scope
- No `WorkflowProfile` registry, picker, or "add workflow" UI.
- No second concrete profile — the seam is left unfurnished (acceptance test: a
  future profile is additive, touching no engine code).
- No relocation of `guided/` into a neutral `staged/` package (Approach C deferred).
- No change to the advisor's monolithic skill pack.
- No scoping of the transform chain solver to per-stage blocks (H1 — accepted as
  whole-skill; optional future optimisation).
- No new backend guided `TurnType` for interpretation review (D12 — reuse freeform).

## 11. Risks / caveats
- **Coordinated multi-site stage addition** (§4.2): ~9 sites in lockstep; import-time
  totality asserts catch omissions loudly.
- **Two duplicated `_ORDER` tuples** (`emitters.py`, `_helpers.py`) — both must be
  updated; appending is safe, mid-insert is not.
- **Silent-orphan fix (B1)** is the most load-bearing backend change; the surfacing
  pass must fire after **every** site-creating commit (source, transform, recipe-apply)
  and cover **all five kinds**, not just prompt_template.
- **Wire view must not render `state.edges`**; it needs **two reads** —
  `get_pipeline_state` (connection-label topology) + `preview_pipeline`
  (`edge_contracts` overlay) — and reads `from`/`to` keys (B2/M1).
- **Advisor must gate the terminal stamp inside the wire branch** (B5/D13), or the 409
  guard dead-ends the revise turn; **every non-CLEAN verdict fails closed** (no manual
  proceed); the re-entry bound only works because the pass counter is **persisted**
  (D16). Size budgets to avoid the 300s timeout.
- **Profile leak (finding 10):** `composer_meta` is copied verbatim on fork — strip the
  profile to empty at the copy seam or the tutorial state leaks into ordinary sessions.
- **Migration:** purge `sessions.db` + bump `SESSION_SCHEMA_EPOCH`; **user secrets are
  destroyed** with it; follow `docs/runbooks/staging-session-db-recreation.md`
  (single-DB path; back up the full `-wal/-shm/-journal` set first).
- **web_scrape recipe (D11)** is new engine work; web_scrape is a **transform** (the
  predicate keys the `inline_blob` source); it must stage the raw-HTML cleanup
  `pipeline_decision` or the deterministic path trips the blocking cleanup contract,
  and record the **audited shield suppression** (`elspeth-abb2cb0931`).
- Preserve the `elspeth-abb2cb0931` prompt-injection-shield suppression markers when
  any LLM-node skill content moves into a stage block.

## 12. Key file map (for the implementation plan)
- Engine (guided package): `composer/guided/protocol.py`, `state_machine.py`,
  `steps.py`, `prompts.py`, `emitters.py`, `chain_solver.py`, `recipe_match.py`,
  `skills/`.
- Recipe catalog (composer top-level, **NOT** under `guided/`): `composer/recipes.py`
  (`RecipeSpec` + `_build_*`; add the web_scrape RecipeSpec keyed on the `inline_blob`
  source per D11/B4).
- Route layer: `sessions/routes/_helpers.py` (dispatcher), `sessions/routes/composer.py`
  (incl. the new guided/tutorial start endpoint, GET /guided, the 409 guard at
  `:2131`, the terminal read at `:2381`).
- Interpretation: `composer/service.py` (`_auto_surface_prompt_template_reviews`
  `:1412`, surface/gate `:2864`, orphan sites `:1376`), `interpretation_state.py`,
  `execution/service.py:514-525`, `interpretation_events` store / `models.py:648`.
- Wire data: `composer/state.py` (`EdgeContract` `:348-368`, `.to_dict` keys `from`/`to`
  `:359-368`, validate `:2680`), `composer/tools/generation.py` (`_execute_preview_pipeline`
  `:1651`, edge_contracts summary `:1685-1692`), `_authoring_validation_payload`
  (`composer/tools/sessions.py:1153`).
- Advisor: `composer/service.py` (`_run_advisor_checkpoint` `:4176`,
  `_build_checkpoint_arguments` `:4108`, `_call_advisor_with_audit` `:3912`,
  `_advisor_blocked_result` `:4130`), `composer/protocol.py:703`,
  `composer/tool_batch.py:653`, `config.py`.
- Tutorial tail/cache: `composer/tutorial_service.py` (`tutorial_model_id` `:784`),
  `preferences/tutorial_cache.py`.
- Migration: `sessions/models.py:117` (`SESSION_SCHEMA_EPOCH`), `sessions/schema.py`,
  `sessions/converters.py:64`, `guided/state_machine.py:41/400`, `app.py:874` (secret
  store on session engine).
- Frontend: `frontend/src/components/tutorial/`, `frontend/src/components/chat/guided/`,
  `frontend/src/types/guided.ts`, `frontend/src/api/client.ts`,
  `frontend/src/stores/interpretationEventsStore.ts`.
