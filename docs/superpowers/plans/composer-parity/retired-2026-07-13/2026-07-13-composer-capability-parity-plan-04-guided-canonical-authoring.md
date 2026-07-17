# Composer Capability Parity Plan 04: Guided Canonical Authoring

> **RETIRED (2026-07-17): DO NOT EXECUTE.** See
> [the current disposition](../2026-07-17-current-plan-disposition.md).

**Goal:** Wire guided-full, guided-staged, and tutorial-profile to the same
canonical planner and commit path as freeform, with complete DAG authoring and
stage-aware intent handling.

**Architecture:** Guided stages collect reviewed conversational facts and
control review order. They call `plan_pipeline()` for topology and
`commit_pipeline_proposal()` for acceptance. They never own topology schema or
materialize a graph from a transform list.

**Depends on:** Plans 01-03.

## File structure

**Create:**

- `src/elspeth/web/composer/guided/stage_subjects.py`
- `src/elspeth/web/composer/guided/deferred_intents.py`
- `tests/integration/web/composer/guided/test_plural_sources_outputs.py`
- `tests/integration/web/composer/guided/test_wrong_stage_deferral.py`
- `tests/integration/web/composer/guided/test_pipeline_proposal_flow.py`
- `tests/integration/web/composer/guided/test_pipeline_proposal_transitions.py`
- `tests/integration/web/composer/guided/test_guided_full_entrypoint.py`
- `tests/unit/web/composer/guided/test_stage_subjects.py`

**Modify:**

- `src/elspeth/web/composer/guided/steps.py`
- `src/elspeth/web/composer/guided/chat_solver.py`
- `src/elspeth/web/composer/guided/emitters.py`
- `src/elspeth/web/composer/guided/state_machine.py`
- `src/elspeth/web/composer/guided/planning.py` — generalize Plan 03's staged adapter across all guided configurations.
- `src/elspeth/web/composer/guided/profile.py`
- `src/elspeth/web/sessions/routes/composer/guided.py`
- `src/elspeth/web/sessions/routes/_helpers.py`
- frontend guided conversation, review, and back/edit components.

## Task 1: Make sources and outputs plural reviewed facts

- [ ] Drive `/guided/start` through two named sources and two named outputs;
  prove stable ids survive edit, restart, reordering, and wire review.
- [ ] Replace singular source staging with `reviewed_sources` and pending intents
  keyed by stable source id.
- [ ] Replace single sink staging with distinctly named `reviewed_outputs`.
- [ ] Preserve each output's plugin options and write-failure policy; remove
  hard-coded `main` and `discard` assumptions.
- [ ] Ensure source/output review never assigns final topology connection labels.
- [ ] Rewind by `{kind, stable_id}` when a reviewed fact changes, invalidate the
  proposal, and retain unaffected reviewed facts.

Run:

```bash
uv run pytest tests/integration/web/composer/guided/test_plural_sources_outputs.py -q
```

## Task 2: Derive stage responsibility from canonical structure

- [ ] Define a closed `StageSubject` vocabulary for source, output, topology,
  and wire-review concerns.
- [ ] Derive plugin subjects from the live catalog's registered kind plus the
  request's intended role. Do not infer kind from a hard-coded plugin-name list.
- [ ] Derive structural subjects from canonical manifest fields: sources and
  source policies; outputs and failure policies; nodes/edges/routes/merge/queue/
  field operations; final wire validation.
- [ ] If one name exists in multiple plugin kinds and intent does not resolve the
  role, ask the operator to disambiguate. Do not guess.
- [ ] Keep “not installed/not permitted” distinct from “valid but belongs at a
  different stage.”

Run:

```bash
uv run pytest tests/unit/web/composer/guided/test_stage_subjects.py -q
```

## Task 3: Implement the wrong-stage wait/back-edit lifecycle

- [ ] When a valid request belongs to a future stage, reply explicitly, for
  example: “That LLM belongs in the transformation stage. Finish the source
  choice first; I have kept this request for that stage.”
- [ ] Persist an ordered `DeferredStageIntent` containing stable intent id,
  receiving/target stages, catalog plugin kind/name when applicable, a redacted
  summary, and summary hash.
- [ ] Do not configure the plugin early, discard the request, advance the stage,
  spend a repair, or call the capability unsupported.
- [ ] Inject deferred intent into the target stage's planner context after
  restart and mark it consumed only when a reviewed fact/proposal covers it.
- [ ] If the responsible stage is already complete, enter or offer stable-id
  back/edit instead of creating an orphan deferral.
- [ ] On proposal rejection/revision, keep the intent pending until the revised
  accepted proposal covers it. On explicit operator cancellation, audit the
  cancellation and remove it.
- [ ] Test source-at-output, sink-at-source, LLM-at-source, transform-at-output,
  ambiguous CSV/JSON names, unavailable plugins, restart, rejection, revision,
  consumption, cancellation, and completed-stage back/edit.

Run:

```bash
uv run pytest tests/integration/web/composer/guided/test_wrong_stage_deferral.py -q
```

## Task 4: Define three guided planner contexts over one implementation

- [ ] Add configuration values for `GUIDED_FULL`, `GUIDED_STAGED`, and
  `TUTORIAL_PROFILE`; these select prompt assembly and interaction context only.
- [ ] Make every configuration invoke the public `plan_pipeline()` from Plan 02
  with the canonical schema, current state, original intent, reviewed facts,
  deferred intents, and structured repair feedback.
- [ ] Add a server-hosted authenticated guided-full entrypoint that exercises
  request parsing, prompt/tool/catalog assembly, proposal creation, candidate
  validation, and audited commit. It is an evaluation surface, not a direct
  planner helper or test-only constructed proposal.
- [ ] Make `/guided/start` and all subsequent staged routes use the same planner
  implementation for topology.
- [ ] Make tutorial call the staged route with a teaching profile; fixed data and
  copy may differ, planner/schema/catalog may not.
- [ ] Add request tracing that records the actual rendered-message hash,
  advertised schema/tool hashes, catalog snapshot hash, surface, and repair
  count without sensitive content.

Run:

```bash
uv run pytest tests/integration/web/composer/guided/test_guided_full_entrypoint.py tests/integration/web/composer/guided/test_pipeline_proposal_flow.py -q
```

## Task 5: Implement the immutable proposal lifecycle

- [ ] `propose`: save a hash-verified proposal after candidate validation, emit
  `PROPOSE_PIPELINE`, and do not publish candidate state.
- [ ] `accept`: under the compose lock, verify echoed draft hash, base content
  hash, and reviewed-anchor hash, then pass `proposal.pipeline` unchanged to
  `commit_pipeline_proposal()`.
- [ ] Capture the arguments at the audited executor boundary and assert deep
  equality with the accepted proposal. Secret resolution/redaction inside the
  executor is outside that comparison.
- [ ] `reject`: record a redacted rejection fact, clear the active proposal, and
  retain operator intent/reviewed facts.
- [ ] `revise`: preserve supersession hash lineage, update only the requested
  intent/facts, and generate a new immutable proposal.
- [ ] `back/edit`: select stable component id, invalidate affected downstream
  review/proposal state, then replan.
- [ ] Stale or mismatched hashes return 409 with no state/audit commit side
  effects beyond the allowlisted rejection event.

Run:

```bash
uv run pytest tests/integration/web/composer/guided/test_pipeline_proposal_transitions.py tests/integration/web/composer/guided/test_proposal_audit_projection.py -q
```

## Task 6: Make wire review authoritative for arbitrary DAGs

- [ ] Render and review the validated `CompositionState`; never reconstruct a
  spine from dialogue facts.
- [ ] Check all produced connections, fan-out/fan-in, routes, failure paths,
  merge/queue policy, guaranteed fields, and output business schemas.
- [ ] Support multiple sources/outputs, gates, aggregation, row expansion,
  queues, coalesces, and explicit error sinks in both backend projection and UI.
- [ ] A wire-review correction returns structured feedback to `plan_pipeline()`;
  it does not mutate the graph behind the proposal hash.

Run:

```bash
uv run pytest tests/integration/web/composer/guided/test_wire_dispatch.py tests/integration/web/composer/guided/test_pipeline_proposal_flow.py -q
```

## Plan 04 completion gate

```bash
uv run pytest \
  tests/unit/web/composer/guided/test_stage_subjects.py \
  tests/integration/web/composer/guided/test_plural_sources_outputs.py \
  tests/integration/web/composer/guided/test_wrong_stage_deferral.py \
  tests/integration/web/composer/guided/test_pipeline_proposal_flow.py \
  tests/integration/web/composer/guided/test_pipeline_proposal_transitions.py \
  tests/integration/web/composer/guided/test_guided_full_entrypoint.py \
  tests/integration/web/composer/guided/test_wire_dispatch.py -q
cd src/elspeth/web/frontend
npm test -- src/components/chat/guided src/stores/sessionStore.guided.test.ts
npm run typecheck
cd ../../../..
uv run ruff check src/elspeth/web/composer/guided src/elspeth/web/sessions/routes/composer/guided.py
git diff --check
```

Plan 04 passes only when all product/evaluation entrypoints use the shared
planner and commit seam, complex guided DAGs are authorable, and every
wrong-stage case waits, rewinds, disambiguates, or errors according to catalog
truth without losing intent.
