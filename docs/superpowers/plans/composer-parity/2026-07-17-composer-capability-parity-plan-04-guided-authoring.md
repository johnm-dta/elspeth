# Composer Capability Parity Plan 04: Guided Authoring and Shared Capability

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:subagent-driven-development` or `superpowers:executing-plans` to
> implement this plan task by task. Steps use checkbox (`- [ ]`) syntax for
> tracking.

**Goal:** Make guided-full, guided-staged, and tutorial use the shared canonical
planner while staged guided authoring supports plural reviewed components,
retains valid wrong-stage intent, and commits only after authoritative wire
review.

**Architecture:** Guided stages own reviewed facts, stable component identity,
conversation order, and pending proposal review; they never own a second graph
builder. Every topology-producing surface calls the same `plan_pipeline()`
implementation and terminal schema. A guided candidate remains a durable
pending proposal through Step 3 and Step 4; only Confirm Wiring prepares,
dispatches, accepts, and consumes mechanically covered deferred intent.

**Tech stack:** Python 3.13, FastAPI, Pydantic, SQLAlchemy, React, TypeScript,
Vitest, pytest.

**Prerequisite:** Plans 01-03 are committed and passing.

---

## Existing foundation to preserve

Do not rebuild work already delivered by Plans 01-03:

- `GuidedSession` schema 8 already stores plural `source_order` /
  `reviewed_sources` and `output_order` / `reviewed_outputs` with UUID identity,
  strict permutations, pending source/output intents, deferred intents, an
  active proposal reference, and a stable edit target.
- `guided/stage_transitions.py` already implements the one-component
  selection/form/review transitions. `guided/steps.py` was deleted and must not
  return.
- `SinkResolved.outputs` is already the canonical plural sink projection. Its
  entries are canonical output definitions, not guided stable ids; keep stable
  ids in `GuidedSession.output_order` and `reviewed_outputs`.
- `guided/planning.py` already binds reviewed plural facts, produces the safe
  proposal projection, defines closed deferred constraint types, and can verify
  constraints mechanically.
- `guided_operations` already supplies leased, fenced, retryable operation
  custody for start/respond/chat/convert/reenter/revert/fork.
- The current Step-3 acceptance path commits too early: it prepares and accepts
  the proposal, clears the active reference, consumes covered deferred intent,
  and only then shows Step 4. Task 6 replaces that lifecycle; do not preserve
  it.

## Hard-cut rule

Task 4 moves guided persistence to schema 9 and the session store to epoch 31.
There is no schema-8 reader, migration, dual request shape, fallback route, or
frontend storage migration. Update active fixtures and tests to schema 9; a
stale pre-release store fails at startup with recreate guidance. Do not change
signed tier allowlists, HMAC metadata, fingerprint baselines, contract
whitelists, release signatures, or artifact manifests.

## Task 1: Add plural component controllers without changing canonical output shape

**Files:**

- Modify: `src/elspeth/web/composer/guided/state_machine.py`
- Modify: `src/elspeth/web/composer/guided/stage_transitions.py`
- Modify: `src/elspeth/web/composer/guided/protocol.py`
- Modify: `src/elspeth/web/composer/guided/emitters.py`
- Modify: `src/elspeth/web/sessions/schemas.py`
- Modify: `src/elspeth/web/sessions/protocol.py`
- Modify: `src/elspeth/web/sessions/service.py`
- Modify: `src/elspeth/web/sessions/routes/composer/guided.py`
- Modify: `src/elspeth/web/frontend/src/types/guided.ts`
- Create: `src/elspeth/web/frontend/src/components/chat/guided/ComponentReviewTurn.tsx`
- Create: `src/elspeth/web/frontend/src/components/chat/guided/ComponentReviewTurn.test.tsx`
- Modify: `src/elspeth/web/frontend/src/components/chat/guided/GuidedTurn.tsx`
- Modify: `src/elspeth/web/frontend/src/components/chat/guided/GuidedTurn.test.tsx`
- Modify: `src/elspeth/web/frontend/src/stores/sessionStore.ts`
- Modify: `src/elspeth/web/frontend/src/stores/sessionStore.guided.test.ts`
- Create: `tests/integration/web/composer/guided/test_plural_sources_outputs.py`

- [ ] **Step 1: Write the failing controller journey**

  Drive the real start/respond API through two named sources and two named
  outputs. Exercise add, edit, exact-permutation reorder, remove, restart,
  proposal rejection, and wire review. Assert the same UUID identifies each
  surviving component throughout and that the final canonical
  `SinkResolved.outputs` remains an ordered sequence of output definitions.

- [ ] **Step 2: Add one closed component-action contract**

  Add a `REVIEW_COMPONENTS` turn and a discriminated action body with these
  exact shapes:

  ```python
  class AddComponentAction(BaseModel):
      action: Literal["add"]
      component_kind: Literal["source", "output"]

  class EditComponentAction(BaseModel):
      action: Literal["edit"]
      target: GuidedEditTargetRequest

  class RemoveComponentAction(BaseModel):
      action: Literal["remove"]
      target: GuidedEditTargetRequest

  class ReorderComponentsAction(BaseModel):
      action: Literal["reorder"]
      component_kind: Literal["source", "output"]
      stable_ids: list[UUID]

  class FinishComponentsAction(BaseModel):
      action: Literal["finish"]
      component_kind: Literal["source", "output"]
  ```

  Emit only server-authored labels, stable ids, current order, and allowed
  actions. Reject duplicates, missing ids, cross-kind ids, and a remove that
  would leave zero reviewed sources or outputs at the point a later stage
  requires one.

- [ ] **Step 3: Implement pure plural transitions**

  Extend `stage_transitions.py` with pure functions named
  `add_component_intent()`, `begin_component_edit()`,
  `remove_reviewed_component()`, and `reorder_reviewed_components()`.
  Add allocates one UUID before the operation attempt and reuses it on retry;
  edit preserves the UUID and user-facing name; reorder accepts only an exact
  permutation; remove deletes the mapping and order entry together. Preserve
  plugin options, source validation policy, output failure policy, and reviewed
  business fields for every unaffected component.

- [ ] **Step 4: Invalidate a pending proposal in the same transaction**

  Add a service command that, under the operation fence and compose lock,
  verifies the current proposal reference, writes `proposal.rejected` with
  reason `superseded`, terminalizes the pending row, applies the reviewed-fact
  mutation, clears proposal/edit references, and persists the rewound guided
  checkpoint in one transaction. A fault at every write boundary must roll
  back the row, event, and checkpoint together.

- [ ] **Step 5: Render and submit the controller**

  Render add/edit/remove/reorder/finish without array indices as authority.
  The frontend sends the exact stable-id action and keeps the existing retry
  operation until the authoritative response settles.

- [ ] **Step 6: Verify and commit**

  ```bash
  uv run pytest tests/integration/web/composer/guided/test_plural_sources_outputs.py tests/unit/web/composer/guided/test_stage_transitions.py -q
  cd src/elspeth/web/frontend
  npm test -- --run src/components/chat/guided/ComponentReviewTurn.test.tsx src/components/chat/guided/GuidedTurn.test.tsx src/stores/sessionStore.guided.test.ts
  npm run typecheck
  cd ../../../..
  uv run ruff check src/elspeth/web/composer/guided src/elspeth/web/sessions
  uv run mypy src/elspeth/web/composer/guided src/elspeth/web/sessions
  git diff --check
  git add src/elspeth/web/composer/guided src/elspeth/web/sessions src/elspeth/web/frontend/src tests/integration/web/composer/guided/test_plural_sources_outputs.py
  git commit -m "feat(composer): add plural guided component controls"
  ```

  Expected: all checks pass and no superseded proposal remains executable.

## Task 2: Create, retain, verify, cancel, and back-edit deferred intent

**Files:**

- Create: `src/elspeth/web/composer/guided/stage_subjects.py`
- Create: `src/elspeth/web/composer/guided/deferred_intents.py`
- Modify: `src/elspeth/web/composer/guided/state_machine.py`
- Modify: `src/elspeth/web/composer/guided/chat_solver.py`
- Modify: `src/elspeth/web/composer/guided/planning.py`
- Modify: `src/elspeth/web/composer/pipeline_planner.py`
- Modify: `src/elspeth/web/composer/service.py`
- Modify: `src/elspeth/web/composer/guided/audit.py`
- Modify: `src/elspeth/web/sessions/routes/composer/guided.py`
- Create: `tests/unit/web/composer/guided/test_stage_subjects.py`
- Create: `tests/unit/web/composer/guided/test_deferred_intents.py`
- Create: `tests/unit/web/composer/guided/test_deferred_intent_coverage.py`
- Modify: `tests/unit/web/composer/test_pipeline_planner.py`
- Create: `tests/integration/web/composer/guided/test_wrong_stage_intent.py`

- [ ] **Step 1: Extract and test the closed subject/constraint boundary**

  Move the existing deferred subject and constraint types out of the large
  state-machine module without changing their schema. Define the stage names as
  `Literal["source", "output", "topology", "wire_review"]`. Resolve plugin
  kind/name from the request-scoped policy catalog; if a permitted name exists
  in several kinds, return an explicit clarification result instead of
  guessing.

- [ ] **Step 2: Add a server-validated deferred-intent action**

  The stage chat solver may propose only this closed result:

  ```python
  @dataclass(frozen=True, slots=True)
  class DeferredIntentAction:
      target_stage: StageName
      catalog_kind: Literal["source", "transform", "sink"] | None
      catalog_name: str | None
      redacted_summary: str
      constraints: tuple[DeferredConstraint, ...]
  ```

  The route validates it against the live catalog and current stage, then
  persists a UUID, receiving/target stages, allowlisted structural facts,
  redacted summary/hash, and the same-session originating user-message id plus
  exact content hash. Raw prose stays only in the private chat row. Invalid or
  unprovable constraints remain pending for explicit edit; they are never
  recorded as covered.

- [ ] **Step 3: Give the common terminal schema optional claims**

  Change the sole terminal schema for every surface to:

  ```json
  {
    "type": "object",
    "properties": {
      "pipeline": {"$ref": "canonical set_pipeline schema"},
      "claimed_deferred_intent_ids": {
        "type": "array",
        "items": {"type": "string", "format": "uuid"},
        "uniqueItems": true
      }
    },
    "required": ["pipeline"],
    "additionalProperties": false
  }
  ```

  Freeform and guided-full pass no eligible ids. Guided-staged/tutorial pass
  only current, message-hash-verified ids. Unknown, duplicate, stale, or
  ineligible claims are malformed output and enter the bounded repair path.
  Remove `tuple(item.intent_id for item in guided.deferred_intents)` from
  `ComposerServiceImpl.plan_guided_pipeline()`; model context is not coverage.

- [ ] **Step 4: Verify claims mechanically before hashing the proposal**

  Implement this boundary in `deferred_intents.py`:

  ```python
  def evaluate_deferred_intent_coverage(
      *,
      candidate: CompositionState,
      reviewed_guided: GuidedSession,
      claimed_intent_ids: tuple[str, ...],
  ) -> tuple[str, ...]:
      claimed = set(claimed_intent_ids)
      verified: list[str] = []
      for intent in reviewed_guided.deferred_intents:
          if intent.intent_id not in claimed:
              continue
          if not all(constraint_holds(candidate, reviewed_guided, constraint) for constraint in intent.constraints):
              raise DeferredIntentClaimError("guided proposal claimed an unproven deferred intent")
          verified.append(intent.intent_id)
      if claimed != set(verified):
          raise DeferredIntentClaimError("guided proposal claimed an unknown deferred intent")
      return tuple(verified)
  ```

  Extract `constraint_holds()` from the current mechanical verifier rather
  than creating a second predicate implementation. Define
  `DeferredIntentClaimError` as a typed invalid-model-output error and map it to
  bounded terminal-tool feedback in `pipeline_planner.py`. Reserve
  `AuditIntegrityError` for corrupted server authority, cross-session rows, or
  message-content hash mismatch; those failures do not enter model repair.

  It returns an ordered subset only when every persisted predicate is true of
  the validated candidate. Re-run it before proposal staging and again before
  final acceptance after re-resolving and hash-checking each originating
  message. Only its return value enters immutable
  `covered_deferred_intent_ids`.

- [ ] **Step 5: Add cancel and passed-stage back/edit**

  Explicit cancellation removes one stable intent and emits the allowlisted
  cancellation event in the same settlement. If the responsible stage has
  passed, enter stable-id back/edit and atomically invalidate downstream review
  or pending proposal state. Rejection, revision, planner repair, timeout, and
  cancellation of the request leave all unconsumed intents pending.

- [ ] **Step 6: Verify and commit**

  ```bash
  uv run pytest \
    tests/unit/web/composer/guided/test_stage_subjects.py \
    tests/unit/web/composer/guided/test_deferred_intents.py \
    tests/unit/web/composer/guided/test_deferred_intent_coverage.py \
    tests/unit/web/composer/test_pipeline_planner.py \
    tests/integration/web/composer/guided/test_wrong_stage_intent.py -q
  uv run ruff check src/elspeth/web/composer src/elspeth/web/sessions/routes/composer/guided.py
  uv run mypy src/elspeth/web/composer src/elspeth/web/sessions
  git diff --check
  git add src/elspeth/web/composer src/elspeth/web/sessions tests/unit/web/composer/guided tests/integration/web/composer/guided/test_wrong_stage_intent.py
  git commit -m "feat(composer): retain verified guided stage intent"
  ```

  Expected: all checks pass; unsupported/policy-denied remains distinct from a
  valid future-stage request.

## Task 3: Render one capability core into every planner surface

**Files:**

- Create: `src/elspeth/web/composer/skills/pipeline_capabilities.md`
- Create: `src/elspeth/web/composer/capability_skill.py`
- Modify: `src/elspeth/web/composer/prompts.py`
- Modify: `src/elspeth/web/composer/guided/prompts.py`
- Modify: `src/elspeth/web/composer/skills/pipeline_composer.md`
- Modify: `src/elspeth/web/composer/guided/skills/base.md`
- Modify: `src/elspeth/web/composer/guided/skills/step_3_transforms.md`
- Modify: `src/elspeth/web/composer/pipeline_planner.py`
- Create: `tests/unit/web/composer/test_capability_skill_identity.py`
- Modify: `tests/unit/web/composer/test_prompts.py`
- Modify: `tests/unit/web/composer/guided/test_skill.py`
- Modify: `tests/unit/web/composer/test_pipeline_planner.py`

- [ ] **Step 1: Write identity and disclaimer failures first**

  Assert that freeform, guided-staged, and tutorial prepend the exact same
  capability-core bytes and advertise the same canonical terminal/discovery
  tool identities. Reject guided text that says gates, queues, forks,
  coalesces, multiple sources, multiple outputs, aggregation, row expansion,
  or structured LLM fields are unavailable. Add guided-full to this matrix in
  Task 5 when its production controller exists.

- [ ] **Step 2: Split capability from interaction policy**

  Move topology capability, discovery order, canonical field semantics,
  structured validation repair, plugin-assistance rules, and
  unsupported-versus-policy language into `pipeline_capabilities.md`.
  `pipeline_composer.md` retains freeform convergence policy. Guided stage
  files retain timing, questions, and presentation policy. Delete the current
  single-spine and gate/fork/merge disclaimers from guided base and Step 3.

- [ ] **Step 3: Build and use a hash manifest from actual call inputs**

  Define one immutable manifest:

  ```python
  @dataclass(frozen=True, slots=True)
  class PlannerCapabilityManifest:
      surface: PlannerSurface
      profile: str
      planner_implementation_id: str
      capability_core_hash: str
      canonical_schema_hash: str
      effective_tool_hash: str
      rendered_prompt_hash: str
  ```

  Compute it from the actual rendered system messages and advertised tools for
  the call, not from constants that can drift independently. Keep
  stage-specific rendered prompt hashes distinct while core/schema/tool hashes
  remain identical.

- [ ] **Step 4: Add canonical drift coverage**

  Enumerate the canonical node/structural fields covered by the core. A new
  canonical node type or field must fail the identity test until both the core
  and parity mapping account for it.

- [ ] **Step 5: Verify and commit**

  ```bash
  uv run pytest \
    tests/unit/web/composer/test_capability_skill_identity.py \
    tests/unit/web/composer/test_prompts.py \
    tests/unit/web/composer/guided/test_skill.py \
    tests/unit/web/composer/test_pipeline_planner.py -q
  uv run ruff check src/elspeth/web/composer
  uv run mypy src/elspeth/web/composer
  git diff --check
  git add src/elspeth/web/composer tests/unit/web/composer
  git commit -m "feat(composer): share canonical capability guidance"
  ```

  Expected: all checks pass and no guided capability disclaimer remains.

## Task 4: Hard-cut the guided operation and wire-review protocol

**Files:**

- Modify: `src/elspeth/web/composer/guided/state_machine.py`
- Modify: `src/elspeth/web/composer/guided/protocol.py`
- Modify: `src/elspeth/web/sessions/models.py`
- Modify: `src/elspeth/web/sessions/protocol.py`
- Modify: `src/elspeth/web/sessions/schemas.py`
- Modify: `src/elspeth/web/sessions/guided_operations.py`
- Modify: `src/elspeth/web/sessions/service.py`
- Modify: `src/elspeth/web/composer/pipeline_custody.py`
- Modify: `src/elspeth/web/frontend/src/types/guided.ts`
- Modify: `src/elspeth/web/frontend/src/api/guidedDecoder.ts`
- Create: `src/elspeth/web/frontend/src/api/guidedDecoder.test.ts`
- Modify: `src/elspeth/web/frontend/src/stores/guidedOperationRetry.ts`
- Modify: `src/elspeth/web/frontend/src/stores/guidedOperationRetry.test.ts`
- Rename: `tests/integration/web/composer/guided/test_schema8_epoch.py` to
  `tests/integration/web/composer/guided/test_schema9_epoch.py`
- Modify: `tests/integration/web/composer/guided/test_guided_operations_schema.py`
- Modify: `tests/integration/web/composer/guided/test_retry_safe_routes.py`
- Modify: `tests/unit/web/sessions/test_guided_operation_requests.py`
- Modify: `tests/unit/web/sessions/test_guided_operations_service.py`
- Modify: `tests/unit/web/sessions/test_schema.py`
- Modify: `tests/unit/web/composer/guided/test_state_machine.py`
- Modify: `tests/unit/web/composer/guided/test_protocol.py`

- [ ] **Step 1: Write the schema-9 and operation-contract failures**

  Require session epoch 31, guided schema 9, operation kind `guided_plan`, a
  pipeline-proposal result locator, and a Step-4 turn that remains bound to the
  pending proposal id/draft hash. Prove schema 8 and epoch 30 fail closed; do
  not add a decoder or migration.

- [ ] **Step 2: Add exact request/result contracts**

  Add:

  ```python
  GuidedOperationKind = Literal[
      "guided_start",
      "guided_respond",
      "guided_chat",
      "guided_convert",
      "guided_reenter",
      "guided_plan",
      "state_revert",
      "session_fork",
  ]

  @dataclass(frozen=True, slots=True)
  class GuidedPipelineProposalResult:
      proposal_id: UUID
      checkpoint_state_id: UUID

  class GuidedPlanRequest(_GuidedOperationRequest):
      intent: str = Field(min_length=1, max_length=4096)

  class StartGuidedRequest(_GuidedOperationRequest):
      profile: object = "live"
      intent: str | None = Field(default=None, min_length=1, max_length=4096)
  ```

  After resolving the server-owned profile, require visible intent for `live`
  and forbid it for `tutorial`. Include the exact intent in the normalized
  request hash. Store proposal results as a distinct closed DB result kind that
  requires both result columns and rejects every other combination.

  Add `request_cancelled` to the closed operation failure vocabulary. If the
  server observes cancellation before atomic proposal/start staging, settle
  that failure honestly. If staging already completed, replay the durable
  result.

- [ ] **Step 3: Make pending proposal authority legal in Step 4**

  Schema 9 permits `active_proposal` only when it is coupled to either the sole
  unanswered trailing Step-3 `PROPOSE_PIPELINE` turn or the sole unanswered
  trailing Step-4 `CONFIRM_WIRING` turn. Make Step-4 payload require
  `proposal_id` and `draft_hash`. Update revert/fork validation to enforce both
  legal shapes and reject orphan, ambiguous, terminal, or cross-checkpoint
  references.

- [ ] **Step 4: Bump the store and frontend retry schema without compatibility code**

  Set `SESSION_SCHEMA_EPOCH = 31`, `GUIDED_SESSION_SCHEMA_VERSION = 9`, and the
  DB CHECKs for the new kind/result shape. Replace the frontend retry key,
  envelope, and request-fingerprint domains with v2 and add `guided_plan`; read
  only v2. Update all active fixtures in one commit.

  Every custody, proposal, status, and result write compares lease token and
  attempt at its transactional write boundary. If the current custody seam
  cannot accept that fence, extend `pipeline_custody.py`; a route-level check
  alone does not fence a late worker.

- [ ] **Step 5: Preserve the signed route-layout boundary**

  The protected tier fingerprints depend on module body positions. Do not move
  existing handlers or edit signed metadata. Keep `post_guided_convert` as the
  final handler defined in `guided.py`. Define `/guided/plan` on a separate
  `APIRouter` in `guided_plan.py`. After the complete
  `post_guided_convert` definition, append a late
  `from .guided_plan import router as guided_plan_router` import (with the
  repository's narrow lint suppression for a load-bearing late import) and then
  `router.include_router(guided_plan_router)`. Neither statement may be inserted
  among existing imports, module statements, or handlers. This leaves every
  existing handler AST position and the final-handler constraint unchanged.

- [ ] **Step 6: Verify and commit**

  ```bash
  uv run pytest \
    tests/integration/web/composer/guided/test_schema9_epoch.py \
    tests/integration/web/composer/guided/test_guided_operations_schema.py \
    tests/integration/web/composer/guided/test_retry_safe_routes.py \
    tests/unit/web/sessions/test_guided_operation_requests.py \
    tests/unit/web/sessions/test_guided_operations_service.py \
    tests/unit/web/sessions/test_schema.py \
    tests/unit/web/composer/guided/test_state_machine.py \
    tests/unit/web/composer/guided/test_protocol.py -q
  cd src/elspeth/web/frontend
  npm test -- --run src/stores/guidedOperationRetry.test.ts src/types/guided.test.ts src/api/guidedDecoder.test.ts
  npm run typecheck
  cd ../../../..
  uv run ruff check src/elspeth/web/composer/guided src/elspeth/web/sessions
  uv run mypy src/elspeth/web/composer/guided src/elspeth/web/sessions
  git diff --check
  git add src/elspeth/web/composer/guided src/elspeth/web/sessions src/elspeth/web/frontend/src tests/integration/web/composer/guided tests/unit/web/composer/guided
  git commit -m "feat(composer): cut guided operations to schema 9"
  ```

  Expected: all checks pass; schema 8 and epoch 30 are not resumable.

## Task 5: Add the production guided-full endpoint and make staged start authoritative

**Files:**

- Create: `src/elspeth/web/sessions/routes/composer/guided_plan.py`
- Modify: `src/elspeth/web/sessions/routes/composer/guided.py`
- Modify: `src/elspeth/web/composer/guided/planning.py`
- Modify: `src/elspeth/web/composer/guided/profile.py`
- Modify: `src/elspeth/web/composer/service.py`
- Modify: `src/elspeth/web/sessions/service.py`
- Modify: `src/elspeth/web/frontend/src/api/client.ts`
- Modify: `src/elspeth/web/frontend/src/api/client.guided.test.ts`
- Modify: `src/elspeth/web/frontend/src/stores/sessionStore.ts`
- Modify: `src/elspeth/web/frontend/src/stores/sessionStore.guided.test.ts`
- Modify: `src/elspeth/web/frontend/src/components/chat/ChatPanel.tsx`
- Modify: `src/elspeth/web/frontend/src/components/chat/ChatPanel.test.tsx`
- Create: `tests/integration/web/composer/guided/test_guided_full.py`
- Create: `tests/integration/web/composer/guided/test_guided_operation_retry.py`
- Create: `tests/integration/web/composer/guided/test_shared_planner_surfaces.py`

- [ ] **Step 1: Write real route tests before the controller**

  Test authenticated `POST /api/sessions/{session_id}/guided/plan` from empty
  and existing canonical states. Assert `PlannerSurface.GUIDED_FULL`, the shared
  planner function identity, common core/schema/tools, one private user-message
  row, one custody chain, one pending proposal, and replay of the same response.
  Hash mismatch returns 409; active lease conflicts; takeover fences the old
  worker; timeout/disconnect before atomic staging leaves no proposal.

- [ ] **Step 2: Implement the focused guided-full controller**

  `guided_plan.py` owns the request lifecycle and returns the ordinary
  `CompositionProposalResponse`. Resolve ownership, rate limit, inflight tally,
  progress registry, provider deadline, disconnect cancellation,
  request-scoped catalog/policy, planner, custody, candidate validation, and
  durable staging on the server. Reserve/claim `guided_plan` before the first
  provider call or custody write, persist the originating intent exactly once,
  and complete the operation with `GuidedPipelineProposalResult`.

  After the provider/custody preparation phase, one fenced service command and
  one database transaction must insert the originating chat row, stage exactly
  one pending proposal/checkpoint, write its creation event, bind the proposal
  result locator, and mark the `guided_plan` operation completed. A crash or
  injected fault cannot expose a pending proposal without a replayable completed
  operation, and takeover cannot create a second proposal. After
  `post_guided_convert`, late-import its separate router and immediately include
  it; do not define another handler in `guided.py` or move that import upward.

- [ ] **Step 3: Make `/guided/start` own the ordinary root intent**

  For `live`, reserve the start operation, persist the exact intent once, bind
  `root_intent_message_id`, and settle the first schema-9 checkpoint together.
  Feed that message into every later staged planner call after rechecking
  session, role, and content hash. Tutorial keeps its fixed server-owned lesson
  seed and supplies no client intent.

- [ ] **Step 4: Remove cold-start `/guided/chat`**

  `GET /guided` hydrates only. The frontend sends the first ordinary guided
  message through `startGuidedSession(intent, operation_id)`, holds the v2 retry
  descriptor across reload/network retry, and uses `/guided/chat` only after a
  guided checkpoint exists. The server rejects cold `/guided/chat` before a
  provider call or durable message write.

- [ ] **Step 5: Prove every surface reaches one planner**

  Guided-staged calls `plan_pipeline()` with reviewed facts and verified
  deferred claims; tutorial calls the same adapter with `TUTORIAL_PROFILE`;
  guided-full calls it with no reviewed/deferred facts. Add guided-full to the
  Task-3 capability manifest matrix and assert no controller owns a second
  planning loop or topology constructor.

- [ ] **Step 6: Verify and commit**

  ```bash
  uv run pytest \
    tests/integration/web/composer/guided/test_guided_full.py \
    tests/integration/web/composer/guided/test_guided_operation_retry.py \
    tests/integration/web/composer/guided/test_shared_planner_surfaces.py \
    tests/unit/web/composer/test_capability_skill_identity.py -q
  cd src/elspeth/web/frontend
  npm test -- --run src/api/client.guided.test.ts src/stores/sessionStore.guided.test.ts src/components/chat/ChatPanel.test.tsx
  npm run typecheck
  npm run lint
  cd ../../../..
  uv run ruff check src/elspeth/web/composer src/elspeth/web/sessions
  uv run mypy src/elspeth/web/composer src/elspeth/web/sessions
  git diff --check
  git add src/elspeth/web/composer src/elspeth/web/sessions src/elspeth/web/frontend/src tests
  git commit -m "feat(composer): add retry-safe guided full planning"
  ```

  Expected: all checks pass and cold guided chat is unreachable.

## Task 6: Keep proposals pending through authoritative arbitrary-DAG wire review

**Files:**

- Modify: `src/elspeth/web/composer/guided/planning.py`
- Modify: `src/elspeth/web/composer/guided/emitters.py`
- Modify: `src/elspeth/web/composer/guided/protocol.py`
- Modify: `src/elspeth/web/composer/guided/state_machine.py`
- Modify: `src/elspeth/web/sessions/schemas.py`
- Modify: `src/elspeth/web/sessions/protocol.py`
- Modify: `src/elspeth/web/sessions/service.py`
- Modify: `src/elspeth/web/sessions/routes/composer/guided.py`
- Modify: `src/elspeth/web/frontend/src/types/guided.ts`
- Modify: `src/elspeth/web/frontend/src/api/guidedDecoder.ts`
- Modify: `src/elspeth/web/frontend/src/api/guidedDecoder.test.ts`
- Modify: `src/elspeth/web/frontend/src/components/chat/guided/ProposePipelineTurn.tsx`
- Modify: `src/elspeth/web/frontend/src/components/chat/guided/ProposePipelineTurn.test.tsx`
- Modify: `src/elspeth/web/frontend/src/components/chat/guided/GuidedTurn.tsx`
- Modify: `src/elspeth/web/frontend/src/components/chat/guided/GuidedTurn.test.tsx`
- Modify: `src/elspeth/web/frontend/src/components/chat/guided/WireStageTurn.tsx`
- Modify: `src/elspeth/web/frontend/src/components/chat/guided/WireStageTurn.test.tsx`
- Modify: `src/elspeth/web/frontend/src/stores/sessionStore.ts`
- Modify: `src/elspeth/web/frontend/src/stores/sessionStore.guided.test.ts`
- Modify: `tests/unit/web/sessions/test_guided_operation_revert_service.py`
- Create: `tests/integration/web/composer/guided/test_arbitrary_dag_review.py`
- Modify: `tests/integration/web/composer/guided/test_pipeline_proposal_reference.py`

- [ ] **Step 1: Write the lifecycle failure first**

  Prove that Step-3 Review Wiring changes only the guided checkpoint and leaves
  the proposal row pending, the canonical composition state uncommitted, and
  deferred intents unconsumed. Prove that Step-4 Confirm Wiring is the only
  action that prepares/dispatches the candidate, writes `proposal.accepted`,
  commits the state, clears the reference, and consumes the mechanically
  covered ids in one transaction.

- [ ] **Step 2: Derive wire review from the pending candidate**

  Load the authoritative proposal and validated candidate; never reconstruct a
  spine from dialogue facts. Project proposal id/draft hash, produced
  connections, fan-out/fan-in, route/failure paths, gate/queue/coalesce/
  aggregation policies, required/guaranteed fields, row cardinality including
  row expansion, output business schemas, structured LLM fields, and validation
  warnings. Every node/edge/source/output target uses the stable ids advertised
  by the exact proposal projection.

- [ ] **Step 3: Change Step 3 from acceptance to review**

  Replace the Step-3 `accept` action with `review_wiring`. Verify the pending
  authority and exact proposal projection, answer the Step-3 turn, advance to
  Step 4, and append a proposal-bound `CONFIRM_WIRING` turn without calling
  `prepare_pipeline_proposal_commit()` or
  `accept_guided_pipeline_proposal()`. Keep `active_proposal` unchanged.

- [ ] **Step 4: Add structured wire correction with immutable supersession**

  Add an exact correction body containing the echoed proposal id/draft hash, a
  stable edit target, and bounded private feedback. Persist feedback once as a
  same-session user chat row and keep only its id/content hash in guided
  authority. Re-plan through the shared planner, mechanically re-evaluate
  deferred claims, and atomically stage a successor with both
  `supersedes_proposal_id` and `supersedes_draft_hash` while rejecting the old
  pending row as `superseded`. Stay in Step 4 and emit a new proposal-bound wire
  turn. A stale/ambiguous target or changed message hash returns 409 with no
  state change.

- [ ] **Step 5: Make Confirm Wiring the sole acceptance transaction**

  Under the compose lock and operation fence, re-load current checkpoint,
  proposal row, creation event, private candidate, projection payload,
  reviewed-fact anchor, current policy/catalog snapshot, and all root/deferred/
  correction messages. Recheck every content hash, mechanically re-evaluate
  claimed deferred intent, prepare one exact dispatch, accept the proposal,
  persist the committed state and answered wire turn, consume only verified
  ids, and complete the guided operation together. Any fault or lease loss
  rolls the whole acceptance back.

- [ ] **Step 6: Update revert/fork and frontend semantics**

  Revert must treat both Step-3 and Step-4 pending references as live authority
  and terminalize any proposal that the rewind invalidates in the same
  transaction. Fork is different: it must leave the source checkpoint and
  source proposal pending and unchanged, while the child checkpoint strips the
  source proposal/edit reference and rewinds to topology review using the
  existing Plan-03 fork sanitation. No child may reference or reject the
  source session's proposal row.

  The UI labels Step 3 as proposal review, exposes Review Wiring rather than
  Accept, renders the richer arbitrary-DAG projection, submits corrections
  against the echoed proposal binding, and exposes Confirm Wiring only when
  server-authored blockers permit it.

  Add SQLite plus environment-gated PostgreSQL races for confirm-versus-
  correction, confirm-versus-revert, and correction-versus-revert. Exactly one
  fenced settlement may win; the loser returns a stable conflict and cannot
  publish, consume, or leave a second pending authority.

- [ ] **Step 7: Verify and commit**

  ```bash
  uv run pytest \
    tests/integration/web/composer/guided/test_arbitrary_dag_review.py \
    tests/integration/web/composer/guided/test_pipeline_proposal_reference.py \
    tests/unit/web/sessions/test_guided_operation_revert_service.py -q
  cd src/elspeth/web/frontend
  npm test -- --run src/components/chat/guided src/stores/sessionStore.guided.test.ts src/api/guidedDecoder.test.ts
  npm run typecheck
  npm run lint
  cd ../../../..
  uv run ruff check src/elspeth/web/composer src/elspeth/web/sessions
  uv run mypy src/elspeth/web/composer src/elspeth/web/sessions
  git diff --check
  git add src/elspeth/web/composer src/elspeth/web/sessions src/elspeth/web/frontend/src tests
  git commit -m "feat(composer): defer guided commit to wire confirmation"
  ```

  Expected: all checks pass; no Step-3 action commits a pipeline.

## Definition of done

- Every topology-producing controller calls the same `plan_pipeline()` object
  and advertises the same canonical terminal/discovery tool identity.
- Ordinary guided start owns the durable root intent; cold guided chat is
  rejected.
- Plural source/output controls preserve stable identity and atomically
  invalidate obsolete pending authority.
- Only mechanically proven deferred claims enter proposal authority, and only
  final acceptance consumes them.
- A guided proposal remains pending through arbitrary-DAG wire review;
  corrections supersede it immutably; Confirm Wiring is the sole commit point.
- Schema 9 / epoch 31 is a hard cut with no schema-8 compatibility path.
