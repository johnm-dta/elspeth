# Composer Capability Parity Plan 04: Guided Authoring and Shared Capability

**Goal:** Make guided-full, guided-staged, and tutorial use the shared planner
with complete canonical capability and stage-aware intent retention.

**Architecture:** Stages collect plural reviewed facts and control conversation
order. They derive a planner context but never construct graph topology. A
shared capability-core document is rendered into freeform and every guided
planner call; stage documents add interaction guidance only.

**Prerequisite:** Plans 01-03 pass.

## Task 1: Make reviewed sources and outputs plural

**Files:**

- Modify: `src/elspeth/web/composer/guided/state_machine.py`
- Modify: `src/elspeth/web/composer/guided/steps.py`
- Modify: `src/elspeth/web/composer/guided/emitters.py`
- Modify: `src/elspeth/web/sessions/routes/composer/guided.py`
- Create: `tests/integration/web/composer/guided/test_plural_sources_outputs.py`

- [ ] Drive the real staged API through two named sources and two named
  outputs. Prove their stable ids survive edit, reordering, restart, proposal
  rejection, and wire review.
- [ ] Implement plural source controller behavior and stable named-output
  identity/editing over the mappings already serialized by schema 8 in Plan 03.
  Preserve the existing plural `SinkResolved.outputs` capability while moving
  its sequence entries to stable ids. Preserve plugin options, source
  validation policy, output failure policy, and user-facing names; do not
  change the persisted checkpoint shape or epoch in this phase.
- [ ] Do not assign final connection labels or edges during source/output
  review. Those are canonical planner output.
- [ ] When a reviewed fact changes, invalidate the active proposal, retain
  unaffected facts, and rewind by `{kind, stable_id}`. Invalidation must call
  Plan 03's atomic transition: reject/supersede the durable pending row and
  clear or replace the checkpoint reference/edit target together. Add a
  regression proving no old pending row remains executable after the edit.

Run:

```bash
uv run pytest tests/integration/web/composer/guided/test_plural_sources_outputs.py -q
```

Expected: PASS.

## Task 2: Retain wrong-stage intent and support back/edit

**Files:**

- Create: `src/elspeth/web/composer/guided/stage_subjects.py`
- Create: `src/elspeth/web/composer/guided/deferred_intents.py`
- Modify: `src/elspeth/web/composer/pipeline_planner.py`
- Modify: `src/elspeth/web/composer/guided/chat_solver.py`
- Modify: `src/elspeth/web/sessions/routes/composer/guided.py`
- Create: `tests/unit/web/composer/guided/test_stage_subjects.py`
- Create: `tests/unit/web/composer/guided/test_deferred_intent_coverage.py`
- Create: `tests/integration/web/composer/guided/test_wrong_stage_intent.py`

- [ ] Define a closed stage-subject vocabulary for source, output, topology,
  and wire-review concerns. Resolve plugin kind from the live request-scoped
  catalog; if a name exists in several kinds, ask rather than guess.
- [ ] For a valid future-stage request, persist an ordered stable intent id,
  receiving/target stages, catalog kind/name when known, redacted summary, and
  summary hash plus the same-session originating user-message id and exact
  message-content hash. Resolve and hash-check the original private text when
  planning; do not duplicate raw prose in
  composer metadata or synthetic audit events. Reply that it has been retained.
  Do not configure it, advance, spend a repair, or call it unsupported.
- [ ] Persist only closed, server-validated constraint predicates that can be
  checked against a canonical candidate: plugin/stable subject presence,
  exact option or numeric values, counts, edge routes, and failure routes.
  Preserve the original message/content hash as authority. If a constraint
  cannot be represented or proven, leave the intent pending for explicit
  back/edit instead of claiming coverage.
- [ ] Let the terminal tool return optional *claimed* deferred-intent ids, but
  treat them as untrusted. Implement a server-owned
  `evaluate_deferred_intent_coverage()` that rejects unknown/stale/message-hash
  mismatches and returns an id only when every persisted predicate is
  mechanically true of the validated candidate. Only those returned ids enter
  immutable `covered_deferred_intent_ids`; merely placing an intent in planner
  context or model output never consumes it. Freeform and guided-full pass no
  eligible ids, and all surfaces retain one identical terminal schema.
- [ ] Inject retained intent into the responsible planner context after
  restart. Mark it consumed only when a reviewed fact or accepted proposal
  covers it. Rejection/revision leaves it pending; explicit cancellation emits
  the allowlisted cancellation event.
- [ ] If the responsible stage has passed, enter stable-id back/edit and
  invalidate downstream review/proposal state instead of making an orphaned
  deferral.
- [ ] Keep “not installed/not permitted” distinct from “valid at another
  stage.” Test source-at-output, sink-at-source, LLM-at-source,
  transform-at-output, ambiguous names, unavailable plugins, restart,
  rejection, revision, consumption, cancellation, and completed-stage edit.
  Include negation, exact numeric/routing constraints, multiple similar intents,
  message/session mismatch, and content-hash mismatch.

Run:

```bash
uv run pytest \
  tests/unit/web/composer/guided/test_stage_subjects.py \
  tests/unit/web/composer/guided/test_deferred_intent_coverage.py \
  tests/integration/web/composer/guided/test_wrong_stage_intent.py -q
```

Expected: PASS.

## Task 3: Render one capability core into every planning surface

**Files:**

- Create: `src/elspeth/web/composer/skills/pipeline_capabilities.md`
- Create: `src/elspeth/web/composer/capability_skill.py`
- Modify: `src/elspeth/web/composer/prompts.py`
- Modify: `src/elspeth/web/composer/guided/prompts.py`
- Modify: `src/elspeth/web/composer/skills/pipeline_composer.md`
- Modify: `src/elspeth/web/composer/guided/skills/base.md`
- Modify: `src/elspeth/web/composer/guided/skills/step_3_transforms.md`
- Create: `tests/unit/web/composer/test_capability_skill_identity.py`

- [ ] Move topology capability, discovery order, canonical field semantics,
  plugin assistance, structured validation repair, and unsupported-vs-policy
  language into `pipeline_capabilities.md`.
- [ ] Make freeform, guided-full, guided-staged, and tutorial prompt assembly
  prepend the exact same core bytes. Stage files may discuss timing, questions,
  and review presentation only.
- [ ] Remove guided gate/queue/fork/coalesce/multi-source disclaimers and the
  “single linear spine” rule.
- [ ] Hash the actual rendered messages and advertised tools per call; expose a
  test manifest with surface/profile, planner implementation id, capability
  core hash, canonical schema hash, effective tool hash, and rendered prompt
  hash.
- [ ] Add drift tests showing a new canonical node type or structural field
  fails capability coverage until the shared core and parity fixture mapping
  account for it.

Run:

```bash
uv run pytest \
  tests/unit/web/composer/test_capability_skill_identity.py \
  tests/unit/web/composer/test_prompts.py \
  tests/unit/web/composer/guided/test_prompts.py -q
```

Expected: PASS; all surfaces report the same core/schema/tool identity while
stage-specific rendered prompt hashes may differ.

## Task 4: Add guided-full and wire all guided configurations

**Files:**

- Modify: `src/elspeth/web/sessions/schemas.py`
- Modify: `src/elspeth/web/sessions/routes/composer/guided.py`
- Modify: `src/elspeth/web/composer/guided/planning.py`
- Modify: `src/elspeth/web/composer/guided/profile.py`
- Modify: `src/elspeth/web/frontend/src/api/client.ts`
- Modify: `src/elspeth/web/frontend/src/api/client.guided.test.ts`
- Modify: `src/elspeth/web/frontend/src/stores/sessionStore.ts`
- Modify: `src/elspeth/web/frontend/src/stores/sessionStore.guided.test.ts`
- Modify: `src/elspeth/web/frontend/src/components/chat/ChatPanel.tsx`
- Modify: `src/elspeth/web/frontend/src/components/chat/ChatPanel.test.tsx`
- Create: `tests/integration/web/composer/guided/test_guided_full.py`
- Create: `tests/integration/web/composer/guided/test_guided_operation_retry.py`
- Create: `tests/integration/web/composer/guided/test_shared_planner_surfaces.py`

- [ ] Add authenticated `POST /api/sessions/{session_id}/guided/plan` accepting
  a bounded client operation id and plain-language intent and returning the
  existing pending proposal response. It must resolve ownership,
  request-scoped catalog/policy/profile, prompt/tools, planner, custody,
  candidate validation, and durable proposal on the server. It is not a
  test-only direct planner call.
- [ ] Wire that endpoint through `get_rate_limiter`,
  `_track_compose_inflight`, the current progress registry, provider deadline,
  and `_cancel_on_client_disconnect` in the same order as the ordinary compose
  routes. A rejected, timed-out, or disconnected request leaves no pending
  proposal or state change unless proposal staging had already completed
  atomically and returned a recoverable pending id.
- [ ] Require a bounded client operation id for guided plan and every guided
  start. Before any provider call or custody write, reserve/claim the
  Plan-03 `guided_operations` row using a normalized request hash and persist
  the user intent exactly once as its originating chat row. Guided-full also
  binds that row to planner provenance; staged start binds it as
  `root_intent_message_id`. Tutorial may omit intent and use its fixed lesson
  seed.
- [ ] Make `POST /guided/start` the production boundary for an ordinary
  guided session's first build request. The frontend may use `GET /guided` only
  to hydrate/resume; `ChatPanel` must send the cold-start intent and a
  retry-stable client operation id through `startGuidedSession()`, then use
  `/guided/chat` only after a guided checkpoint exists. Persist the pending
  client operation id until the response is settled so a network retry/reload
  does not mint a different operation. Reject cold-start `/guided/chat` on the
  server instead of silently bypassing operation reservation and root-intent
  binding. Update the API, store, ChatPanel, ordinary-mode, retry, and tutorial
  tests together.
- [ ] Give retries durable semantics: the same operation id and request hash
  returns the existing completed proposal/start result; a hash mismatch returns
  409; an active lease conflicts without starting another provider call; and an
  expired lease may be taken over only with an audit event and incremented
  attempt count. Do not claim provider exactly-once behavior. Instead guarantee
  that retries cannot create duplicate durable chat rows, ready blobs, or
  proposals, using the operation row, originating message id, custody key, and
  proposal settlement together. Persist failed/cancelled status honestly and
  test recovery at every boundary. Every late write must carry and compare the
  current lease token and attempt; losing a lease fences the old worker out of
  custody, proposal, status, and result writes.
- [ ] Feed the durable root intent into every staged planning context so the
  parity harness starts all three ordinary surfaces from the same request.
- [ ] Make guided-staged call the same `plan_pipeline()` with reviewed facts and
  deferred intent at topology planning. Make tutorial call the staged adapter
  with `TUTORIAL_PROFILE`; fixed lesson input/copy may differ, planner/schema/
  discovery/commit may not.
- [ ] Accept guided-full proposals through the same Plan 02 proposal endpoint.
  Accept guided-staged/tutorial through the same internal commit adapter.
- [ ] Add tests that capture the function identity/call boundary and fail if a
  controller grows a second planning loop or topology constructor.

Run:

```bash
uv run pytest \
  tests/integration/web/composer/guided/test_guided_full.py \
  tests/integration/web/composer/guided/test_guided_operation_retry.py \
  tests/integration/web/composer/guided/test_shared_planner_surfaces.py -q
cd src/elspeth/web/frontend
npm test -- --run \
  src/api/client.guided.test.ts \
  src/stores/sessionStore.guided.test.ts \
  src/components/chat/ChatPanel.test.tsx
cd ../../../..
```

Expected: PASS.

## Task 5: Make arbitrary-DAG review authoritative

**Files:**

- Modify: `src/elspeth/web/composer/guided/planning.py`
- Modify: `src/elspeth/web/composer/guided/emitters.py`
- Modify: `src/elspeth/web/sessions/routes/composer/guided.py`
- Modify: `src/elspeth/web/frontend/src/components/chat/guided/WireStageTurn.tsx`
- Create: `tests/integration/web/composer/guided/test_arbitrary_dag_review.py`

- [ ] Derive review from the validated candidate `CompositionState`; never
  reconstruct a spine from dialogue facts.
- [ ] Review produced connections, fan-out/fan-in, routes, failure paths,
  merge/queue policy, required/guaranteed fields, and output business schemas.
- [ ] Cover multiple sources/outputs, gates, aggregation, row expansion,
  queues, coalesces, explicit error sinks, and structured LLM fields in backend
  and UI projections.
- [ ] Send a wire correction as structured planner feedback and produce a new
  immutable proposal with `supersedes_draft_hash`; never mutate arguments
  behind an accepted hash.
- [ ] Have each proposal carry the ordered deferred-intent ids that its exact
  candidate covers according to the Task-2 server resolver. Re-run the
  mechanical predicates and validate originating message hashes against the
  current checkpoint before staging and again before acceptance.
  Revision/rejection leaves the intents pending; only acceptance consumes the
  covered ids, through the Plan-03 atomic proposal + guided-state settlement.
- [ ] Validate an edit target semantically against the referenced proposal:
  `{kind, stable_id}` must resolve exactly once and remain bound to the echoed
  proposal id/draft hash. A syntactically valid stale or ambiguous target
  returns 409 without changing review state.

Run:

```bash
uv run pytest \
  tests/integration/web/composer/guided/test_arbitrary_dag_review.py \
  tests/integration/web/composer/guided/test_pipeline_proposal_reference.py -q
cd src/elspeth/web/frontend
npm test -- --run src/components/chat/guided src/stores/sessionStore.guided.test.ts
npm run typecheck
cd ../../../..
uv run ruff check src/elspeth/web/composer src/elspeth/web/sessions/routes/composer/guided.py
uv run mypy src/elspeth/web/composer src/elspeth/web/sessions
git diff --check
```

Expected: all commands exit 0.

**Definition of done:** Every guided controller reaches the same shared planner,
proposal store, and commit seam as freeform; staged dialogue preserves valid
early intent and uses stable back/edit; and arbitrary canonical DAGs are
reviewed without reconstructing a linear topology.
