# Tutorial staged recut — design

- **Date:** 2026-06-22
- **Status:** Approved (brainstorm complete) — ready for implementation planning
- **Branch:** `worktree-tutorial-staged-recut` (off `release/0.7.0` @ `4bbb0624b`)
- **Target version:** 0.7.0 (pre-release; no backward-compat / no feature flag)
- **Supersedes:** the single-pass "big-bang" first-run tutorial designed in
  `docs/composer/ux-redesign-2026-05/04-first-run-tutorial.md` (esp. line ~232,
  "the user describes the whole pipeline at once; multi-step is what guided mode
  is for") and `21-phase-4-hello-world-tutorial.md`.

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
(per-stage skill blocks, the real lever on #1/#3/#4) and a *flow/UX*
decomposition (a staged wizard that shows the logic, #2/#3).

## 2. Key discovery that shapes the design

The staged wizard already exists. `src/elspeth/web/composer/guided/` is a shipped,
audited staged source → sink → recipe-match → transforms wizard:

- **Per-stage rule blocks already carved out:** `guided/skills/base.md`
  (cross-cutting) + `step_1_source.md`, `step_2_sink.md`, `step_2_5_recipe_match.md`,
  `step_3_transforms.md`. `guided/prompts.py:load_step_chat_skill(step)` injects
  **base + exactly one stage's playbook** ("scopes the LLM's awareness to just the
  step the user is on"). This is the fragility fix, already built.
- **Backend stage state machine:** `GuidedStep` (`guided/protocol.py:96-102`),
  `step_advance` (`guided/state_machine.py:508-554`), per-step widgets, a visible
  stepper.
- **A deterministic recipe-match short-circuit** (`step_2_5`) that reaches a
  completed pipeline in ≤9 clicks, <30s, with **zero LLM calls**.
- **Per-stage interpretation review** is already a guided turn type.

The advisor reads a wholly separate skill source (the monolithic
`composer/skills/pipeline_composer.md` via `build_system_prompt`), so it is
already decoupled from the per-stage blocks.

## 3. Decision

**Co-restructure both (Approach A — generalise guided in place):** the existing
`composer/guided/` state machine becomes *the* staged workflow engine,
parameterised by a `WorkflowProfile`. **The tutorial is a specific instance of
that engine** (profile = canonical seed + coaching + recipe-match + bookends);
live guided mode is the engine with the **empty** (canonical default) profile.

We **do not** relocate guided into a neutral `staged/` package now (that is
Approach C — a mechanical move available later for free; the profile boundary is
identical). We **do not** build a profile registry, a profile picker, or any
"add workflow" UI — only the plumbing (the profile parameter + one wired
instance). A future profile (e.g. "deep research") is *write-a-profile-object +
wire its entry point*, touching no engine code. That additivity is the acceptance
test for the profile boundary.

### Decisions log (resolved during brainstorm)

| # | Decision | Choice |
|---|----------|--------|
| D1 | Tutorial ↔ guided relationship | **Co-restructure both** — one engine, tutorial = instance |
| D2 | Refactor depth | **Approach A** — generalise guided in place (C = extract `staged/` deferred) |
| D3 | Stage order | **source → sink → transform** (matches operator narration AND guided's both-ends-pinned field-reconciliation rationale) |
| D4 | Wiring | Its own **rule block** AND its own **user-facing stage** (4th stage: `wire`) |
| D5 | Advisor pack | Stays **big-bang monolith** in both end-gate and per-phase escape (already decoupled) |
| D6 | Advisor firing | **Auto whole-pipeline END sign-off** (after wire, before run) + **per-phase on-demand "go to advisor"** escape when a stage is stuck |
| D7 | Staged end-gate polarity | **Blocking** (match freeform fail-closed); re-emit a "revise" turn on flag/unavailable |
| D8 | Profile scope | **Plumbing only** — no registry/picker/menu; one wired instance (tutorial); empty profile = live guided |
| D9 | Frontend | Tutorial **drives the real coached guided stepper** (not an embedded tutorial-only stepper) |
| D10 | Migration | **In-place** for 0.7.0; remove big-bang components; no feature flag |

## 4. Architecture

```
        ┌─────────────────────────────────────────────┐
        │   Staged workflow engine (today's            │
        │   composer/guided/ state machine)            │
        │   reads a  WorkflowProfile  — hardcodes      │
        │   nothing tutorial- or guided-specific       │
        └─────────────────────────────────────────────┘
                 ▲                          ▲
        EMPTY profile               TutorialProfile
        = live guided               = instance #1
        (today's behaviour)         (canonical seed, coaching,
                                     recipe-match, bookends)
```

`WorkflowProfile` is internal plumbing. The engine consumes it; the tutorial
constructs the one concrete profile at its entry point; the empty profile is a
**canonical value, not `None`**, that reproduces today's live-guided behaviour.

### 4.1 Stage flow

`GuidedStep` gains a fifth member **appended after transforms** (append is safe —
it only adds an ordinal; a mid-insert would renumber the wire protocol and is
forbidden):

| # | Stage | Elicits | Rule block |
|---|-------|---------|------------|
| 1 | Source | "where the data comes from" | `step_1_source.md` (exists) |
| 2 | Sink | "where it's going" | `step_2_sink.md` (exists) |
| 2.5 | *recipe-match* | deterministic short-circuit, zero-LLM | `step_2_5_recipe_match.md` (exists) |
| 3 | Transform | "what happens to it" | `step_3_transforms.md` (exists) |
| 4 | **Wire** *(new)* | "see how the pieces connect" + confirm | `step_4_wire.md` (**new**) |

The tutorial profile wraps this with **welcome** and **graduation** bookends
(frontend); the existing **run → audit** tail is unchanged.

Source + sink are pinned first (steps 1–2) so step 3 authors transforms against a
**known sink contract** — the field-reconciliation has both endpoints to validate
against. The wire stage makes that reconciliation **visible** instead of silent.

### 4.2 Adding the wire stage — coordinated touchpoints (verified)

Mechanically simple in the pure stepper, but it must hit all of these in lockstep
(import-time asserts catch omissions, which is desirable):

- `guided/protocol.py:96-102` — add `STEP_4_WIRE` to the `GuidedStep` StrEnum.
- `guided/protocol.py:169-192` — add the step to `_LEGAL_TURN_MATRIX`; add a new
  `TurnType` (e.g. `CONFIRM_WIRING`) to the enum (`:16-24`) plus `_REQUIRED_KEYS`
  (`:200-218`) and `_NESTED_SHAPES` (`:243-253`), both **total** over `TurnType`.
- `guided/prompts.py:43-58` — add the step to `_STEP_FILE_NAMES` and
  `_STEP_PLAYBOOK_ORDER` and create `guided/skills/step_4_wire.md`, or the
  import-time asserts (`:64-73`) crash.
- `guided/emitters.py:428-432` **and** `sessions/routes/_helpers.py:3674-3678` —
  append the step to **both** duplicated `_ORDER` ordinal tuples; add a
  `build_step_4_wire_turn` emitter.
- `guided/state_machine.py:508-554` — add the `STEP_4_WIRE` branch + `_advance_step_4`;
  change `_advance_step_3`'s accept path to advance to `STEP_4` instead of letting
  the route terminate.
- `sessions/routes/_helpers.py:3339` (route `STEP_3` block) — redirect chain-accept
  away from immediate `COMPLETED` to emit the wire turn; add the `STEP_4_WIRE`
  dispatch branch that carries the `COMPLETED` terminal (move terminal-stamping out
  of `handle_step_3_chain_accept`).
- **Both** completion seams must redirect: `handle_step_3_chain_accept`
  (`steps.py:390-406`) **and** `handle_step_2_5_recipe_apply` (`steps.py:262-274`,
  the recipe short-circuit) stamp `COMPLETED` today; both must hand off to wire
  first, or the recipe path skips wiring.
- `sessions/routes/composer.py:1366-1414` — add the `STEP_4_WIRE` rebuild branch in
  the GET `/guided` re-fetch dispatcher.
- `frontend/src/types/guided.ts:35-39` — add the step string to the hand-written
  `GuidedStep` TS union (and the `TurnType` union at `:18-28`).

### 4.3 WorkflowProfile threading (greenfield — no profile concept exists today)

- Add a frozen, persisted `profile: WorkflowProfile` field to `GuidedSession`
  (`state_machine.py:257-332`): bump `GUIDED_SESSION_SCHEMA_VERSION` **5 → 6**, add
  the field with a default, add strict `to_dict`/`from_dict` lines (direct-key, no
  `.get()`), and delete the pre-v6 sessions DB on deploy (per the dataclass
  contract; standing operator grant for ELSPETH DB deletes).
- Parameterise `GuidedSession.initial(profile)` (`state_machine.py:342-351`).
- Thread the profile into `_initial_composition_state_with_guided_session`
  (`_helpers.py:2183-2208`) and `build_initial_step_1_turn` (`emitters.py:51`) for
  the entry-seed / pre-filled-source-vs-empty decision.
- Gate recipe-match toggle (STEP_2_5 dispatch, `_helpers.py:3053-3161`), advisor
  checkpoints (`solve_chain` call sites `_helpers.py:3112/3352`), and welcome/
  graduation copy on profile flags.

`WorkflowProfile` fields (initial): `entry_seed` (canonical prompt + pre-filled
source vs empty), `coaching` (per-stage copy, on/off), `advisor_checkpoints`
(on/off), `recipe_match` (on/off), `bookends` (welcome/graduation). The default
(empty) profile sets all to the live-guided behaviour.

## 5. Backend mechanics

### B1 — Per-stage interpretation review (fixes a latent silent-orphan bug)

**Latent bug (verified):** the freeform fail-closed orphan gate
(`service.py:_missing_pending_interpretation_review_sites`, `:1376`) runs **only**
inside the freeform no-tool finalize loop (`_try_terminate_no_tools`, `:2565`).
Guided commits each step by calling `_execute_*` tools directly (`steps.py`) and
**never enters that loop**. So a guided step 3 that commits an `llm` node creates
real interpretation sites (`llm_prompt_template`, `vague_term`, `llm_model_choice`)
that are **surfaced to no one** and only fail at *run* time with
`UnresolvedInterpretationPlaceholderError` (`execution/service.py:514-525`) and
**zero pending events** — the staging-500 class. The recut hits this the moment a
stage adds an LLM transform, so the fix is in-scope.

**Fix / design:** add an explicit **per-stage surfacing pass** in the guided
dispatcher after each `StepHandlerResult`, **scoped to the transform-commit
boundary** (where LLM nodes first appear). Reuse the freeform primitives —
`_auto_surface_prompt_template_reviews` (`service.py:1412`) and the surface-and-
resolve pair (`service.py:2864`) — refactored into a session-aware helper the
guided route can call (it needs `sessions_service` + `session_id`, both present in
the route). Each site becomes a **resolvable review card** on the partial pipeline.

Polarity: per-stage review is **surface-and-resolve (advisory)**, matching the
`vague_term` doctrine; the existing **run-time gate remains the hard backstop**, so
we do not re-implement fail-closed at step commit. Source-only / source+sink
partial states yield zero sites, so we fire the pass only at transform-commit
(`invented_source` surfaces naturally at the source stage). Watch: the narrow
raw-HTML cleanup contract (`composition_review_contract_error` inside
`_execute_set_pipeline`, `sessions.py:657`) is already BLOCKING for
`web_scrape → field_mapper` raw-field drops — the tutorial's web_scrape chain must
stage a `pipeline_decision` or it hard-fails the commit (400), not a card.

### B2 — Wire stage data model (no new backend / endpoint)

The end-to-end field-flow data already exists in `CompositionState.validate()`:

- `edge_contracts` (`state.py:348-368`, emitted `:1683-1692` / `:1830-1839`):
  `{from_id, to_id, producer_guarantees, consumer_requires, missing_fields,
  satisfied}` — exactly the per-edge field-flow rows a wire view needs.
- `semantic_contracts` (parallel, Phase-1 scoped to `line_explode` / `web_scrape`).
- Already serialized by `preview_pipeline` (`generation.py:1651`) and
  `get_pipeline_state`'s validation payload (`sessions.py:1153-1162`). Source
  columns for the leftmost producer = `GuidedSession.step_1_result.observed_columns`.

**Hard constraint (the one trap):** wiring is carried by **named connection
labels** (`source.on_success` / `node.input` / `node.on_success` / `routes` /
`fork_to`), **not** `EdgeSpec` objects. Guided passes `edges=[]` (`steps.py:359`),
and freeform edges only back-write routing when targeting a sink
(`transforms.py:544`). The wire view **reconstructs topology from connection labels
and overlays `edge_contracts`** — it must never render `state.edges` directly.

**"Confirm wiring" does not commit routing** (the step handlers already wired it).
It (1) gates progression to `COMPLETED`/YAML on `validate().is_valid` (zero blocking
field-contract errors, all `edge_contracts.satisfied`), and (2) lets the user accept
a reconciliation — insert a `field_mapper` or relax a consumer/sink schema to
`flexible` — via existing `upsert_node` / `set_pipeline` tools, making the
chain-solver's silent auto-drop explicit. Honest-gap rendering: coalesce/fork nodes
skip `edge_contracts` (render "not statically checkable"); shape-changer produced-
fields may be unavailable mid-edit (don't present as authoritative until
`validate()` succeeds). The canonical tutorial pipeline has none of these.

### B3 — Advisor wiring (end sign-off + per-phase escape)

The staged flow completes at two seams (`handle_step_2_5_recipe_apply`,
`handle_step_3_chain_accept`) that funnel through one route chokepoint,
`post_guided_respond` (`composer.py:2381`, right after `terminal = guided.terminal`)
— the attach point for the automatic END sign-off.

- **Automatic END sign-off (after wire, before run):** invoke the existing
  `_run_advisor_checkpoint(phase='end')` (`service.py:4176`) +
  `_build_checkpoint_arguments(phase='end')` (`service.py:4108`). It is state-driven,
  non-raising, and already loads the **big-bang monolith** via
  `build_system_prompt(self._data_dir)` (`service.py:3954`) — wholly decoupled from
  the per-stage skills. **Polarity: BLOCKING** (D7) — a flagged/unavailable advisor
  re-emits a "sign-off found issues, please revise" guided turn (re-open the
  transform/wire stage with findings) instead of allowing the run. (The tutorial's
  recipe-match canonical pipeline should return CLEAN, so the block rarely fires on
  the learner path.)
- **Per-phase on-demand "go to advisor":** ride the existing
  `ControlSignal.REQUEST_ADVISOR` (today a no-op re-prompt at step 3 only,
  `_helpers.py:3342`) — repoint it to the whole-pipeline checkpoint with a
  "structurally complete enough to review" guard, and extend it to all stages. Keep
  the trust-tier path correct: operator-triggered escapes must not route through the
  Tier-3 `_validate_advisor_arguments`, and backend checkpoints must not consume
  unvalidated user text.

**Structural change required (verified):** `_run_advisor_checkpoint` /
`_build_checkpoint_arguments` are private on `ComposerServiceImpl` and absent from
the `ComposerService` Protocol (`protocol.py:703`); the staged dispatcher holds no
service handle (only model/temperature/seed). Add a public protocol method
(e.g. `async def run_signoff_checkpoint(state, session_id, recorder, progress) ->
AdvisorCheckpointVerdict`) delegating to the private method, and thread the
composer handle into the route/dispatcher.

**Budget/latency:** each advisor call adds ~`composer_advisor_timeout_seconds`
(default 60s); the staging harness already times out at 300s. The tutorial's
recipe-match path means its only frontier round-trip is the single end sign-off.
Size `composer_advisor_checkpoint_max_passes` (checkpoint budget),
`composer_advisor_max_calls_per_compose` (escape-hatch budget),
`composer_advisor_timeout_seconds`, and `composer_timeout_seconds`; decide whether
the end call is awaited inline vs streamed via progress.

Also correct stale prose: `tools/_dispatch.py:129` still says the advisor is
"Disabled by default" — contradicts the mandatory-advisor reality.

## 6. Run / audit / cache tail

### C1 — Unchanged
`run_tutorial_pipeline` (`tutorial_service.py:82`, `POST /api/tutorial/run`) is
purely post-compose: it runs the persisted latest `CompositionStateRecord`,
agnostic to how it was composed. **Run → audit → graduation survive as-is**, given
the staged flow still lands one final assembled state (it does).

### C2 — Cache: one contained change
The cache exists to skip the expensive **engine run**, which still happens once at
the end. Fix: `tutorial_model_id` (`tutorial_service.py:784-821`) currently folds
the **freeform** `pipeline_composer.md` hash into `model_id`; the staged tutorial is
built by the **per-stage** skills, so `model_id` must fold the **staged skill
hashes** (`base.md` + `step_*.md`) instead, or the cache won't invalidate when a
stage block is edited. The canonical pipeline becomes *more* cacheable (recipe-match
produces a fixed shape with zero LLM calls; only the run-time rating is
nondeterministic, which the cache freezes). Keep posting the canonical seed token at
Turn-4 purely as the cache key (the profile carries it; the user no longer types
it). The topology-match guard (`_state_matches_cached_topology`) already tolerates
compositional nondeterminism (compares only the ordered `(NodeType, plugin)`
sequence).

## 7. Frontend (D9 — coached real stepper)

- **Survives:** `welcome` bookend and the `run → audit → graduation` tail.
- **Replaced:** `describe` + `showBuilt` (big-bang compose + 3-assumption batch
  review) → the staged guided walk (source / sink / transform / wire), each with
  its own per-stage interpretation review.
- **Folds in:** the old `graph` turn is subsumed by the **wire** stage.
- **Collapses:** Turn-6 `mode` choice becomes a graduation affordance ("keep
  building in guided / try freeform") rather than an up-front abstract pick.
- **Mechanism:** the tutorial enters guided mode on the tutorial session with the
  `TutorialProfile`, rendering the **real** `GuidedTurn` stepper (the actual product
  UI) with tutorial chrome (coaching copy, progress, bookends) layered on — the
  strongest answer to "kill the magic box" and maximal reuse of audited code.
- Remove `TutorialTurn2Describe` / `TutorialTurn2bShowBuilt`. `tutorialMachine.ts`
  retains `welcome` + the run/audit/graduation tail; the compose phase is delegated
  to the guided stepper.

## 8. Migration (D10)
In-place for 0.7.0 (pre-release; no tech debt). The `GuidedSession` v5→v6 bump
already forces a sessions-DB delete on deploy. Keep the canonical "cool government
pages" artifact so the validated end-to-end pipeline still applies. No long-lived
feature flag — the big-bang path retires with the version.

## 9. Testing

- **Backend (TDD, load-bearing first):**
  - Per-stage surfacing pass at the transform-commit boundary — an LLM node
    committed in a stage produces a **resolvable card**, not a run-time 500 (the B1
    fix).
  - Wire-stage handler + the `validate().is_valid` confirm gate; topology
    reconstruction from connection labels (not `state.edges`).
  - Blocking END sign-off from the staged completion seam (the revise loop).
  - `WorkflowProfile` schema-v6 strict round-trip (`to_dict`/`from_dict`).
  - Recipe-match short-circuit → completion redirected **through** the wire stage.
- **Frontend:** extend `composer-guided.spec.ts` for the wire stage; rewrite
  `tutorial.spec.ts` to the staged flow (the 3-assumption batch gate becomes
  per-stage reviews); rewrite the staging reliability harness
  (`tutorial-reliability.staging.spec.ts`) for the staged path.
- **Parity guard:** assert the **empty profile reproduces today's live-guided
  behaviour** — the engine stays parity-clean; all tutorial difference lives in the
  profile.

## 10. Non-goals / out of scope
- No `WorkflowProfile` registry, picker, or "add workflow" UI.
- No second concrete profile (no `DeepResearchProfile`) — the seam is left
  unfurnished. Acceptance test: a future profile is additive (write a profile +
  wire an entry point), touching no engine code.
- No relocation of `guided/` into a neutral `staged/` package (Approach C deferred;
  the profile boundary makes it a later mechanical move).
- No change to the advisor's monolithic skill pack.

## 11. Risks / caveats
- **Coordinated multi-site stage addition** (§4.2): ~9 sites in lockstep; the
  import-time asserts (`prompts.py`, `protocol.py` totality) catch omissions loudly.
- **Two duplicated `_ORDER` tuples** (`emitters.py`, `_helpers.py`) — both must be
  updated; appending is safe, mid-insert is not.
- **Silent-orphan fix (B1)** is the single most load-bearing backend change; getting
  the surfacing-pass scope wrong (firing on source/sink stages) flags not-yet-wired
  nodes.
- **Wire view must not render `state.edges`** (B2) — the biggest design trap.
- **Advisor protocol/handle plumbing (B3)** — a new public method + threading the
  handle, not a one-liner; size budgets to avoid reintroducing the 300s timeout.
- **Schema v6 → sessions-DB delete** on deploy.
- Preserve the `elspeth-abb2cb0931` prompt-injection-shield suppression markers when
  any LLM-node skill content moves into a stage block.

## 12. Key file map (for the implementation plan)
- Engine: `composer/guided/protocol.py`, `state_machine.py`, `steps.py`,
  `prompts.py`, `emitters.py`, `chain_solver.py`, `skills/`.
- Route layer: `sessions/routes/_helpers.py` (dispatcher), `sessions/routes/composer.py`.
- Interpretation: `composer/service.py` (`_auto_surface_prompt_template_reviews`
  `:1412`, surface/gate `:2864`, orphan sites `:1376`), `interpretation_state.py`,
  `execution/service.py:514-525`.
- Wire data: `composer/state.py` (`edge_contracts` `:348-368`, validate `:2676`),
  `composer/tools/sessions.py` (`get_pipeline_state` `:1088`), `generation.py:1651`.
- Advisor: `composer/service.py` (`_run_advisor_checkpoint` `:4176`,
  `_build_checkpoint_arguments` `:4108`, `_call_advisor_with_audit` `:3912`),
  `composer/protocol.py:703`, `composer/tool_batch.py:653`, `config.py`.
- Tutorial tail/cache: `composer/tutorial_service.py`, `preferences/tutorial_cache.py`.
- Frontend: `frontend/src/components/tutorial/` (machine + turns),
  `frontend/src/components/chat/guided/`, `frontend/src/types/guided.ts`.
