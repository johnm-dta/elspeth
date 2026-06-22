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
  **CSV sources only** (`recipe_match.py:186` `is_csv`; all recipes hardcode
  `source plugin == csv`, `recipes.py:219/314/469`). It does **not** fire for the
  canonical web_scrape seed today; see §3/D11 (we add a web_scrape recipe).
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
| **D11** | Canonical compose path | **Add a web_scrape recipe** (+ predicate) so the canonical pipeline composes deterministically (zero-LLM) via recipe-match; also generalises recipe-match to web-scrape guided users |
| **D12** | Per-stage review surfacing | **Reuse the freeform `interpretation_events` store** — NO new backend guided `TurnType`; the per-stage surfacing pass writes events the existing UI renders |
| **D13** | Advisor terminal ordering | END sign-off is a **pre-terminal gate inside the wire-commit branch**; CLEAN stamps COMPLETED, FLAGGED/UNAVAILABLE leaves terminal None and re-emits the wire turn with findings; `STEP_4_WIRE` is re-enterable (bounded by `composer_advisor_checkpoint_max_passes`) |
| **D14** | Stage set ownership | The stage SET is a **global engine property** (the frozen-total `GuidedStep` enum), NOT a `WorkflowProfile` field; live guided also gains the wire stage (intended improvement, not a parity regression) |
| **D15** | Schema migration mechanism | **Purge** (not migrate) AND bump `SESSION_SCHEMA_EPOCH` 22→23 so boot fail-closes loudly, converting the silent lazy per-row HTTP 500 into an actionable boot guard |

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

**Fix / design:** add an explicit **per-stage surfacing pass** in the guided
dispatcher after each `StepHandlerResult`, **scoped to the transform-commit
boundary** (where LLM nodes first appear). Reuse the freeform primitives —
`_auto_surface_prompt_template_reviews` (`service.py:1412`) and the surface-and-
resolve pair (`service.py:2864`) — refactored into a session-aware helper the
guided route can call. **These write `interpretation_events` rows surfaced via the
existing `interpretationEventsStore`/`InterpretationReviewTurn` UI (D12) — NOT a
new guided `TurnType`.** Source-only / source+sink partial states yield zero sites,
so fire the pass only at transform-commit (`invented_source` surfaces at the source
stage). Polarity: per-stage review is **surface-and-resolve (advisory)**; the
existing run-time gate remains the hard backstop, so we do not re-implement
fail-closed at step commit. Watch: the raw-HTML cleanup contract
(`composition_review_contract_error` inside `_execute_set_pipeline`,
`sessions.py:657`) is already BLOCKING for `web_scrape → field_mapper` raw-field
drops — the canonical pipeline must stage a `pipeline_decision` or the commit
hard-fails (400). The new web_scrape recipe (D11) must encode that cleanup
correctly so the deterministic path stays clean.

### B2 — The wire stage's data model (corrected)

No new backend/endpoint needed for inspection — but the read surface is
**`preview_pipeline`, not `get_pipeline_state`** (M1 correction). The field-flow
data is `CompositionState.validate().edge_contracts` (+ `semantic_contracts`),
built by `_authoring_validation_payload` (`sessions.py:1153-1162`) and surfaced
**only** by `_execute_preview_pipeline` (`composer/tools/generation.py:1651`;
edge_contracts in its summary at `composer/tools/generation.py:1685-1692`).
`_execute_get_pipeline_state` (`sessions.py:1088-1125`) returns only
`sources/nodes/outputs/edges/metadata/version/inspection` and runs no `validate()`.

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

- **CLEAN** → stamp `COMPLETED` as today.
- **FLAGGED** → do **not** stamp terminal; re-emit the wire turn carrying the
  advisor `findings_text` (terminal stays `None`, so `/guided/respond` accepts the
  next response and the 409 guard stays correct and untouched).
- **UNAVAILABLE** (distinct from FLAGGED — do not conflate) → **soft re-emit**: same
  revise turn, "advisor could not be reached; review manually and re-confirm to
  proceed." This avoids a hard dead-end on transient advisor-transport failure and
  matches the advisory/surface-and-resolve polarity used elsewhere; it deliberately
  does **not** reuse freeform's fail-closed-non-runnable `_advisor_blocked_result`
  for the staged learner path. (The run-time gate remains the hard backstop.)

`STEP_4_WIRE` is **re-enterable** (self-loop, bounded by
`composer_advisor_checkpoint_max_passes`). **Per-phase on-demand "go to advisor":**
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

Add a `RecipeSpec` + predicate so the canonical web_scrape pipeline composes
deterministically. Today `recipe_match.py:186` (`is_csv`) gates every recipe on
`source.plugin == csv`, and all three catalog recipes hardcode csv
(`composer/recipes.py:219/314/469`); `match_recipe` returns `None` for web_scrape
and the dispatcher falls through to the live chain solver (`_helpers.py:3107`,
`solve_chain_with_auto_drop`). Add: (1) a `web_scrape` predicate in
`recipe_match.py` (source plugin == web_scrape, with the expected sink/required-
fields shape), and (2) a `RecipeSpec` in `composer/recipes.py` whose
`_build_*` deterministically emits the canonical web_scrape → llm-rate →
field_mapper(raw-cleanup) → jsonl pipeline, including the `pipeline_decision`
staging the raw-HTML cleanup needs (so it passes the B1 blocking cleanup contract).
This generalises recipe-match to web-scrape guided users too (a co-restructure
win), and is what makes the §4.1 "zero-LLM canonical compose" claim true.

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

1. `composer_model`.
2. The **staged skill hashes** (`base.md` + `step_*.md`) — not the freeform skill —
   so editing a stage block invalidates the cache.
3. A **content hash of the recipe registry** (`composer/recipes.py` / the canonical
   `RecipeSpec` + `_build_*`). Under D11 the recipe **deterministically authors the
   cached pipeline including option-level content** (provider, model,
   prompt_template, response_field, schema mode, output format). The topology-match
   guard (`_state_matches_cached_topology`, `:713-732`) compares only the ordered
   `(NodeType, plugin)` sequence and is **option-blind by design** — so an
   option-only recipe edit would otherwise serve a stale cached YAML as the Tier-1
   audit artifact. Keep the guard option-blind; guarantee option fidelity by
   **keying** every deterministic input instead.

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
- **Mechanism:** the tutorial calls the new start endpoint (§4.3) to enter guided
  mode on the tutorial session with the `TutorialProfile`, then renders the **real**
  `GuidedTurn` stepper with tutorial chrome (coaching copy, progress, bookends)
  layered on. Remove `TutorialTurn2Describe` / `TutorialTurn2bShowBuilt`.
  `tutorialMachine.ts` retains `welcome` + the run/audit/graduation tail.

## 8. Migration (D10, D15 — corrected runbook)

**Purge, do not migrate** (pre-release delete-stale-DB policy; operator holds a
standing grant to delete ELSPETH session DBs). The original "the v5→v6 bump already
forces a sessions-DB delete on deploy" claim was **wrong**: the `GuidedSession`
blob lives in the `composition_states.composer_meta` JSON column (no new SQL
column), so `GUIDED_SESSION_SCHEMA_VERSION` 5→6 is enforced **lazily** by
`GuidedSession.from_dict` (`state_machine.py:400-403`) as a per-row **HTTP 500** when
a stale guided session is opened — *not* a boot crash. Runbook:

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
5. **Backup:** copy `sessions.db` (+ `-wal`/`-shm`) aside before deletion (reversible
   within the deploy window).
6. **Verification:** assert `PRAGMA user_version == SESSION_SCHEMA_EPOCH` on the new
   DB, plus a smoke test that creating + running a fresh guided/tutorial session
   reaches `COMPLETED` without a 500.

Keep the canonical "cool government pages" artifact so the validated end-to-end
pipeline still applies. No long-lived feature flag.

## 9. Testing

### 9.1 Test cases (TDD, load-bearing first)
- **Backend:**
  - Per-stage surfacing pass at transform-commit — an LLM node committed in a stage
    produces a **resolvable `interpretation_events` card** (D12), not a run-time 500.
  - The new **web_scrape recipe** (D11): predicate matches the canonical shape;
    `_build_*` emits the deterministic pipeline incl. the raw-HTML `pipeline_decision`
    so it passes the blocking cleanup contract; recipe-apply redirects **through** the
    wire stage (not straight to COMPLETED).
  - Wire stage: topology reconstructed from connection labels (not `state.edges`);
    `validate().is_valid` confirm gate; re-validate + re-surface after a
    `field_mapper`/schema-relax reconciliation (B6).
  - **Advisor END sign-off branch matrix (B3/D13):** CLEAN → COMPLETED; FLAGGED →
    re-emit wire turn with findings, terminal stays None; UNAVAILABLE → soft re-emit;
    timeout; budget-exhausted (`composer_advisor_checkpoint_max_passes`). Assert the
    pre-terminal ordering (no 409 dead-end).
  - **Per-phase `REQUEST_ADVISOR` escape:** dispatch from each stage; the trust-tier
    guard (operator-triggered escapes do NOT pass through Tier-3
    `_validate_advisor_arguments`; checkpoints do not consume unvalidated user text);
    a client-tamper case.
  - **Cache identity (C2):** editing a stage block, OR the canonical recipe’s
    option-level content, invalidates the cache; `_state_matches_cached_topology`
    tolerance; single-profile seed-discriminator assertion.
  - `WorkflowProfile` schema-v6 strict round-trip; a **pre-v6 GuidedSession is
    rejected** on load; the `SESSION_SCHEMA_EPOCH` 22→23 boot guard fires.
- **Frontend:**
  - Extend `composer-guided.spec.ts` for the wire stage; rewrite `tutorial.spec.ts`
    to the staged flow (the 3-assumption batch gate becomes per-stage reviews via the
    interpretation store); rewrite `tutorial-reliability.staging.spec.ts`.
  - **Bridging (D9):** assert the `TutorialProfile` drives the **real** `GuidedTurn`
    stepper (the start endpoint seeds the profile; GET /guided reads it), not a
    tutorial-only stepper.
- **Parity guard (D14):** the empty profile differs from the tutorial profile ONLY
  in the profile-gated dimensions (entry_seed, coaching, advisor_checkpoints,
  recipe_match, bookends) — NOT byte-identical to the pre-recut 4-step flow (the enum
  now has 5 members for all profiles).

### 9.2 Gates / verification commands
- `uv run ruff check src/ tests/ scripts/ examples/ elspeth-lints/src/`
- `uv run ruff format --check src/ tests/ scripts/ examples/ elspeth-lints/src/`
- `uv run mypy src/ elspeth-lints/src/`
- elspeth-lints (composer + plugin_contract/immutability/audit_evidence rulesets
  touched by wire/profile/recipe changes), e.g.
  `PYTHONPATH=elspeth-lints/src uv run python -m elspeth_lints.core.cli check --rules 'composer/*' --root src/elspeth`
- targeted `pytest` over the new guided/tutorial/recipe test files
- frontend: `npm run typecheck`, `npm test -- --run` (vitest), `npm run build`,
  Playwright `npm run test:e2e` (+ `test:e2e:staging` for the reliability harness)
- `wardline scan . --fail-on ERROR` (exit 0 clean / 1 gate tripped / 2 tool error) —
  B1/B3/recipe touch externally-fed advisor/user-text trust boundaries. See the
  `wardline-gate` skill at `.agents/skills/wardline-gate/SKILL.md` (do not cite
  `docs/agents.md`; it is absent here).

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
- **Silent-orphan fix (B1)** is the most load-bearing backend change; scope it to the
  transform-commit boundary.
- **Wire view must not render `state.edges`**; must call `preview_pipeline` (not
  `get_pipeline_state`) and read `from`/`to` keys (B2/M1).
- **Advisor must gate the terminal stamp inside the wire branch** (B5/D13), or the 409
  guard dead-ends the revise turn; size budgets to avoid the 300s timeout.
- **Migration:** purge `sessions.db` + bump `SESSION_SCHEMA_EPOCH`; **user secrets are
  destroyed** with it; back up first (§8).
- **web_scrape recipe (D11)** is new engine work; it must encode the raw-HTML cleanup
  `pipeline_decision` or the deterministic path trips the blocking cleanup contract.
- Preserve the `elspeth-abb2cb0931` prompt-injection-shield suppression markers when
  any LLM-node skill content moves into a stage block.

## 12. Key file map (for the implementation plan)
- Engine: `composer/guided/protocol.py`, `state_machine.py`, `steps.py`,
  `prompts.py`, `emitters.py`, `chain_solver.py`, `recipe_match.py`, `recipes.py`,
  `skills/`.
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
