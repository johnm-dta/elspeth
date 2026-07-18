# Composer Capability Parity Plan 03: Guided Schema and Protocol Replacement

**Goal:** Replace transform-only guided persistence and wire contracts with a
single current schema that references canonical durable pipeline proposals.

**Architecture:** Guided schema 8 persists reviewed facts, deferred intent, and
a verified `GuidedProposalRef`; exact executable arguments remain in the
existing private `composition_proposals` row. `PROPOSE_PIPELINE` carries only a
redacted graph projection and stable edit targets. Session epoch 29 makes the
pre-release replacement fail at startup instead of lazily inside one route.

**Prerequisite:** Plans 01 and 02 pass. This is an atomic backend/frontend
cutover: do not leave an active v7 decoder or a reachable chain proposal arm.

## Task 1: Define the current guided checkpoint

**Files:**

- Modify: `src/elspeth/web/composer/guided/state_machine.py`
- Modify: `src/elspeth/web/composer/guided/resolved.py`
- Create: `tests/unit/web/composer/guided/test_schema8_state.py`
- Modify: `tests/unit/web/composer/guided/test_state_machine.py`

- [x] Add failing round-trip, strict-key, strict-enum, recursive-immutability,
  and wrong-version tests for schema 8.
- [x] Replace `ChainProposal`, `step_3_proposal`, and `step_3_edit_index` with:

```python
GUIDED_SESSION_SCHEMA_VERSION = 8


@dataclass(frozen=True, slots=True)
class GuidedProposalRef:
    proposal_id: UUID
    draft_hash: str
    base: ProposalBase
    reviewed_anchor_hash: str
    covered_deferred_intent_ids: tuple[str, ...]
    creation_event_schema: Literal["pipeline_proposal_created.v1"]
    supersedes_proposal_id: UUID | None = None
    supersedes_draft_hash: str | None = None


@dataclass(frozen=True, slots=True)
class ComponentTarget:
    kind: Literal["source", "node", "edge", "output"]
    stable_id: str
```

The session stores `reviewed_sources` and `reviewed_outputs` mappings keyed by
server-generated stable ids, pending source/output intents keyed the same way,
ordered `DeferredStageIntent` records (including originating message id and a
closed tuple of mechanically checkable plugin/option/count/routing/failure
constraints),
`active_proposal: GuidedProposalRef | None`, and
`active_edit_target: ComponentTarget | None`. It also stores an optional
same-session `root_intent_message_id` rather than duplicating the raw initial
request in composer metadata. It does not embed canonical arguments or another
graph model. `GuidedProposalRef.base` reuses Plan 02's tagged `AbsentBase |
PresentBase` union: a new pipeline has an explicit absent base, while a present
base always carries both state id and composition-content hash. It is never a
nullable or wildcard base.

- [x] Preserve current closed decoding, exact types, non-negative counters,
  terminal-state rules, and workflow-profile validation. Reject every
  `schema_version != 8`; do not add a converter. The entire guided composer
  metadata object and every guided request/response DTO use strict closed
  decoding (`extra="forbid"` / exact keys), while unrelated fields in the
  general session envelope retain their existing compatibility posture.
- [x] Cover every guided stage, pending proposal, stable edit target, deferred
  intent, completed/exited state, and process restart.
- [x] Include multi-source/multi-output reordering and stable-id retention in
  the schema-8 round trip now. Plan 04 adds controller behavior over this final
  shape; it must not alter persisted schema 8 after the epoch-29 cutover.

Run:

```bash
uv run pytest \
  tests/unit/web/composer/guided/test_schema8_state.py \
  tests/unit/web/composer/guided/test_state_machine.py -q
```

Expected: PASS.

Task 1 is accepted as the structural schema-8 contract. The Plan 03 cutover is
still an atomic cohort: active routes and frontend callers remain intentionally
unsupported until Tasks 2 through 6 replace the schema-7 protocol.

## Task 2: Allocate the pre-release session boundary

**Files:**

- Modify: `src/elspeth/web/sessions/models.py`
- Modify: `src/elspeth/web/sessions/schema.py`
- Modify: `src/elspeth/web/sessions/protocol.py`
- Modify: `src/elspeth/web/sessions/service.py`
- Modify: `src/elspeth/web/sessions/schemas.py`
- Modify: `src/elspeth/web/sessions/routes/composer/guided.py`
- Modify: `src/elspeth/web/sessions/routes/composer/state.py`
- Modify: `src/elspeth/web/sessions/routes/sessions.py`
- Modify: `src/elspeth/web/frontend/src/types/guided.ts`
- Modify: `src/elspeth/web/frontend/src/api/client.ts`
- Modify: `src/elspeth/web/frontend/src/stores/sessionStore.ts`
- Modify: `docs/runbooks/staging-session-db-recreation.md`
- Modify: `docs/runbooks/aws-ecs-deployment.md`
- Modify: `docs/runbooks/ansible-ubuntu-deployment.md`
- Modify: `docs/guides/sharing-pipelines.md`
- Modify: `tests/unit/web/sessions/test_interpretation_events_table.py`
- Modify: `tests/unit/web/sessions/test_blob_inline_resolutions_schema.py`
- Create: `tests/integration/web/composer/guided/test_schema8_epoch.py`
- Create: `tests/integration/web/composer/guided/test_guided_operations_schema.py`
- Create: `tests/unit/docs/test_staging_session_recreation_policy.py`

- [ ] First write a test that opens an epoch-28 session database and expects
  the existing actionable stale-schema error.
- [ ] Bump `SESSION_SCHEMA_EPOCH` from 28 to 29 and extend its history comment:
  schema 29 invalidates guided v7 composer metadata and chain proposal state
  and adds the guided-operation retry table below.
- [ ] Add `guided_operations` to the session store with unique
  `(session_id, operation_id)`, a closed operation kind and status, normalized
  request hash, lease token/expiry, originating message id, optional
  proposal/result ids, attempt count, and timestamps. Bound operation ids and
  store no raw user intent. Implement atomic reserve/claim, completed-result
  lookup, failure settlement, and audited expired-lease takeover consistently
  on SQLite and PostgreSQL. This table is part of epoch 29, not a later lazy
  migration.
- [ ] Put the stable client `operation_id` in the actual strict request DTO,
  route call, frontend API call, and store action for start, respond, chat,
  convert-to-guided, re-enter-guided, revert, and fork. The frontend creates one
  id per user action and reuses it for transport retries; it does not generate a
  fresh id after an ambiguous response.
- [ ] Define replay/re-entry behavior mechanically: the same operation id and
  request hash returns the exact stored completed response; an active attempt
  rejoins/polls the same operation; an expired lease may be taken over with an
  incremented attempt; a terminal failed operation returns the same safe failure
  and an intentional new try needs a new id. Reuse with a different request hash
  or operation kind is a 409 integrity conflict.
- [ ] Fence every post-provider durable mutation with the current
  `(session_id, operation_id, lease_token, attempt)`. Lease takeover, custody
  reservation/finalization, proposal staging, operation-status update, and
  result settlement all run under the same-session lock and compare that tuple
  in their write transaction. A stale worker performs no durable write after
  losing the lease. Add a race where worker A expires, worker B takes over, and
  A completes last; only B may stage or settle one proposal/result.
- [ ] Apply the reservation, replay, lease, and stale-worker tests to all seven
  mutating surfaces, including provider-free revert/fork paths. Revert and fork
  still take the compose lock and operation fence; idempotency does not weaken
  their proposal/blob cleanup rules.
- [ ] Do not change `SQLITE_SCHEMA_EPOCH`; this feature does not alter the
  landscape/audit database.
- [ ] Update exact epoch assertions in current tests and verify a fresh SQLite
  and PostgreSQL session store reports epoch 29.
- [ ] Update the staging recreation runbook for both supported session-store
  shapes: drain/stop, resolve and confirm the exact session store, retain only
  sanitized diagnostics when useful, recreate, restart, verify epoch 29 and an
  empty session list, then start a fresh guided session.
- [ ] Keep the existing pre-release policy: no in-place migration, old-source
  switch, compatibility reader, or supported database restore. If startup
  fails, keep staging drained, fix current code, recreate fresh state, and
  retry.
- [ ] Add a docs test that rejects active restore/downgrade instructions and
  checks the epoch and guided-schema reason are current.
- [ ] Update the active AWS ECS, Ansible Ubuntu, and pipeline-sharing documents
  from session epoch 28 to 29 in the same change. Their schema probes and
  troubleshooting instructions must agree with the current constant and the
  authoritative recreation runbook.

Run:

```bash
uv run pytest \
  tests/integration/web/composer/guided/test_schema8_epoch.py \
  tests/integration/web/composer/guided/test_guided_operations_schema.py \
  tests/unit/web/sessions/test_engine.py \
  tests/unit/docs/test_staging_session_recreation_policy.py -q
uv run pytest tests/testcontainer/web/test_schema_probe_postgres.py -q
```

Expected: PASS; epoch 28 fails before serving requests and a fresh store starts
at epoch 29.

## Task 3: Replace the proposal protocol

**Files:**

- Modify: `src/elspeth/web/composer/guided/protocol.py`
- Modify: `src/elspeth/web/composer/guided/emitters.py`
- Modify: `src/elspeth/web/sessions/schemas.py`
- Modify: `src/elspeth/web/sessions/routes/_helpers.py`
- Create: `tests/unit/web/composer/guided/test_propose_pipeline_protocol.py`
- Modify: `tests/unit/web/composer/guided/test_protocol.py`

- [ ] Add `TurnType.PROPOSE_PIPELINE` with a closed payload containing
  `proposal_id`, `draft_hash`, server-generated redacted summary/rationale,
  allowlisted structured blockers, redacted graph/node/output summaries, and
  stable component targets. Never expose model-authored rationale verbatim.
- [ ] Replace `accepted_step_index` and `edit_step_index` in
  `GuidedRespondRequest` / `TurnResponse` with `proposal_id`, `draft_hash`, and
  `edit_target`, and add the required stable `operation_id`. Accept/reject/revise
  must echo both proposal id and draft hash.
- [ ] Update legal-turn maps, exact payload keys, response validation,
  redaction/summary allowlists, audit emission, GET rebuild, and guided dispatch
  together. Add exhaustive enum tests so a new turn cannot be silently omitted.
- [ ] Reject stale/mismatched proposal ids or hashes with 409 before mutation.
  Reject malformed stable ids with 400. GET rebuild encountering a stale edit
  target or any legacy edit index returns an integrity conflict; it does not
  clear, guess, or fall back to an array position.

Run:

```bash
uv run pytest \
  tests/unit/web/composer/guided/test_propose_pipeline_protocol.py \
  tests/unit/web/composer/guided/test_protocol.py \
  tests/unit/web/sessions/routes/test_request_advisor_escape.py -q
```

Expected: PASS.

## Task 4: Integrate the durable proposal reference and audit projection

**Files:**

- Create: `src/elspeth/web/composer/guided/planning.py`
- Modify: `src/elspeth/web/composer/guided/audit.py`
- Modify: `src/elspeth/web/sessions/service.py`
- Modify: `src/elspeth/web/sessions/protocol.py`
- Modify: `src/elspeth/web/sessions/routes/composer/guided.py`
- Modify: `src/elspeth/web/sessions/routes/_helpers.py`
- Create: `tests/integration/web/composer/guided/test_pipeline_proposal_reference.py`
- Create: `tests/integration/web/composer/guided/test_proposal_audit_projection.py`

- [ ] Add a minimum staged adapter that first claims/fences the guided
  operation, loads the expected predecessor, and preallocates the proposal id
  and one new checkpoint-state id. Compute that future checkpoint's composition
  content hash (which excludes guided composer metadata) and pass
  `PresentBase(checkpoint_id, content_hash)` into
  `plan_pipeline()` so the immutable proposal and draft hash are sealed exactly
  once against the state acceptance will see.
- [ ] After planning, use one session-service staging transaction to recheck
  the operation fence and that the expected predecessor id/content hash is
  still current, then atomically write the preallocated state (including the
  emitted pending turn and verified `GuidedProposalRef`), durable proposal row,
  creation event, and operation result. Reject predecessor drift without
  resealing or persisting the draft. Emit `PROPOSE_PIPELINE` only after commit.
  Do not create a second metadata-only state while pending, and never leave a
  proposal row without its guided checkpoint reference or publish candidate
  topology. Test that base/draft hash are computed once, predecessor drift
  rejects staging, and acceptance observes the checkpoint named by
  `PresentBase`.
- [ ] Persist and verify the complete proposal reference/store contract:
  proposal/draft/base/anchor hashes, ordered covered-deferred ids, creation
  event schema, predecessor/supersession links, and closed lifecycle events.
  The private proposal row plus its sole authoritative `proposal.created` event
  remain reconstruction authority; the guided ref is a verified safe anchor,
  not an alternate event log.
- [ ] Rebuild GET responses by loading the same-session pending row and
  recomputing draft/base/anchor hashes. If an expected, same-session row was
  explicitly rejected or superseded, acquire the compose lock, clear the
  active reference/edit target, and deterministically return the controller to
  topology planning. Missing, altered, unexpectedly terminal, or cross-session
  rows remain hard integrity conflicts rather than empty proposals.
- [ ] On accept, let the route acquire the current compose lock once, call the
  Plan 02 lock-assuming preparation, and pass the cleared/advanced guided
  metadata into the same atomic state + proposal settlement transaction. On
  acceptance, validate the proposal's ordered
  `covered_deferred_intent_ids` against the current checkpoint and consume
  exactly those ids in that same settlement transaction. On reject, atomically
  terminalize the pending row and clear the reference. On revise, atomically
  terminalize the old row as rejected/superseded, create and reference its
  immutable successor, and record the supersession link; the old proposal must
  never remain executable. Rejection/revision preserves deferred intent and
  reviewed facts until a successor is accepted.
- [ ] Store only an allowlisted redacted proposal projection in guided payload
  and event surfaces. Add canaries for raw inline content, credentials,
  resolved secrets, raw validation text, and raw provider errors across
  proposed, failed, revised, rejected, accepted, and restored cases.
- [ ] Preserve the current fail-closed field-mapper guarantee through the shared
  planner: a secret-bearing structured LLM candidate may advertise typed output
  fields only when its validation probe constructs successfully. Expected
  config-probe failure must abstain/warn and block any downstream field mapper
  that requires unproven fields; it must never infer pass-through guarantees or
  accept the proposal by fallback.

Run:

```bash
uv run pytest \
  tests/integration/web/composer/guided/test_pipeline_proposal_reference.py \
  tests/integration/web/composer/guided/test_proposal_audit_projection.py -q
```

Expected: PASS; exact arguments exist only in the private proposal row and the
guided checkpoint holds only the verified reference.

## Task 5: Replace frontend proposal state and renderer

**Files:**

- Create: `src/elspeth/web/frontend/src/components/chat/guided/ProposePipelineTurn.tsx`
- Create: `src/elspeth/web/frontend/src/components/chat/guided/ProposePipelineTurn.test.tsx`
- Modify: `src/elspeth/web/frontend/src/components/chat/guided/GuidedTurn.tsx`
- Modify: `src/elspeth/web/frontend/src/components/chat/guided/InspectAndConfirmTurn.tsx`
- Modify: `src/elspeth/web/frontend/src/components/chat/guided/MultiSelectWithCustomTurn.tsx`
- Modify: `src/elspeth/web/frontend/src/components/chat/guided/SchemaFormTurn.tsx`
- Modify: `src/elspeth/web/frontend/src/components/chat/guided/SingleSelectTurn.tsx`
- Modify: `src/elspeth/web/frontend/src/components/chat/guided/WireStageTurn.tsx`
- Modify: `src/elspeth/web/frontend/src/types/guided.ts`
- Modify: `src/elspeth/web/frontend/src/types/guided.test.ts`
- Modify: `src/elspeth/web/frontend/src/types/interpretation.test.ts`
- Modify: `src/elspeth/web/frontend/src/stores/sessionStore.ts`
- Modify: `src/elspeth/web/frontend/src/stores/sessionStore.guided.test.ts`
- Modify: `src/elspeth/web/frontend/src/test/guided-fixtures.ts`
- Modify: `src/elspeth/web/frontend/src/api/client.guided.test.ts`
- Modify: `src/elspeth/web/frontend/src/components/tutorial/TutorialGuidedShell.test.tsx`
- Modify: `src/elspeth/web/frontend/src/components/chat/ChatPanel.test.tsx`
- Modify: the colocated tests for every guided response component listed above
- Delete: `src/elspeth/web/frontend/src/components/chat/guided/ProposeChainTurn.tsx`
- Delete: `src/elspeth/web/frontend/src/components/chat/guided/ProposeChainTurn.test.tsx`

- [ ] Render the proposal using the existing full-DAG/wire-review visual
  primitives, with node/output/route summaries and stable edit controls.
- [ ] Submit proposal id and draft hash for every action. Display 409 as a stale
  proposal that must be regenerated; never retry acceptance with old data.
- [ ] Close the TypeScript union over `propose_pipeline` and add an exhaustive
  renderer assertion. `GuidedTurn` is a closed discriminated union decoded from
  the wire before entering the store; production guided API/store/renderer code
  may not use `unknown` casts to admit an unvalidated payload.
- [ ] Test forks, fan-in, multiple sources/outputs, gates, queues,
  aggregations, error routes, revise, reject, stale hash, keyboard navigation,
  focus, and tutorial passive controls.
- [ ] Add explicit `ChatPanel` tests for active/reloading/stale/error proposal
  states and tutorial-shell tests proving the passive profile renders the same
  closed `propose_pipeline` payload without enabling forbidden actions.

Run:

```bash
cd src/elspeth/web/frontend
npm test -- --run \
  src/components/chat/guided/ProposePipelineTurn.test.tsx \
  src/components/chat/guided/GuidedTurn.test.tsx \
  src/stores/sessionStore.guided.test.ts
npm run typecheck
```

Expected: PASS.

## Task 6: Make fork/revert safe, then delete the old path

**Files:**

- Modify: `src/elspeth/web/sessions/service.py`
- Modify: `tests/unit/web/sessions/test_fork.py`
- Modify: `src/elspeth/web/sessions/routes/sessions.py`
- Modify: `src/elspeth/web/sessions/routes/composer/state.py`
- Modify: `src/elspeth/web/blobs/protocol.py`
- Modify: `src/elspeth/web/blobs/service.py`
- Modify: `tests/unit/web/blobs/test_service.py`
- Create: `tests/integration/web/composer/guided/test_schema8_fork_revert.py`
- Create: `tests/unit/web/composer/guided/test_no_chain_authoring_path.py`
- Delete: `src/elspeth/web/composer/guided/chain_solver.py`
- Delete: `src/elspeth/web/sessions/_guided_solve_chain.py`
- Delete chain-only code from: `src/elspeth/web/composer/guided/steps.py`
- Delete chain-only code from: `src/elspeth/web/composer/guided/emitters.py`
- Delete chain-only code from: `src/elspeth/web/sessions/routes/_helpers.py`
- Delete chain-only code from: `src/elspeth/web/sessions/routes/composer/guided.py`
- Modify chain-era references/comments in:
  `src/elspeth/web/composer/guided/_discovery.py`,
  `src/elspeth/web/composer/guided/chat_solver.py`,
  `src/elspeth/web/composer/guided/errors.py`,
  `src/elspeth/web/composer/guided/prompts.py`,
  `src/elspeth/web/sessions/_guided_step_chat.py`, and
  `src/elspeth/web/frontend/src/components/tutorial/tutorialMachine.ts`

- [ ] Extend the current fork metadata preparation so the child preserves
  copied reviewed facts and deferred intent, strips the tutorial profile as it
  does today, clears `active_proposal` / `active_edit_target`, and rewinds a
  proposal/wire stage to topology planning. It must not copy a proposal row or
  retain parent session/blob ids.
- [ ] Build a source-message to child-message id map for every copied message
  referenced by `root_intent_message_id` or a deferred intent. Remap those ids
  inside the same fork transaction and fail the fork atomically if a referenced
  message is outside the copied slice. Never leave child metadata pointing at a
  parent chat row.
- [ ] Reuse the current fork blob-rewrite map for every blob id inside copied
  reviewed source facts. If a reviewed fact refers to a blob that is not copied
  and rebound to the child, trigger the compensating cleanup/archive path
  rather than preserve the parent id or return an active child.
- [ ] Preserve the current compensating fork boundary: `fork_session()` first
  commits the child session/messages/state; the route then copies blobs,
  rewrites reviewed-fact blob ids from the returned `blob_map`, and saves the
  rewritten child state. Do not claim filesystem copies participate in the
  session transaction.
- [ ] Fence fork's message map, blob map, rewrite, and result settlement with
  its stable operation id. Replays return the same child; interruption uses the
  compensating cleanup/archive path and can never expose a second or partially
  mapped child.
- [ ] Add an idempotent blob-service cleanup operation for that post-commit
  phase. If blob copy, id rewrite, or rewritten-state persistence fails, clean
  every copied child blob and archive the child using the existing rollback
  helper. If cleanup itself faults, emit the existing audit/metric signal and
  leave the blobs bound only to the archived, unusable child for retention
  reconciliation. Test failures after each post-commit boundary and prove no
  active child retains a parent blob id or partially rewritten metadata.
- [ ] Make the revert route acquire the compose lock before it resolves or
  writes state. A revert always terminalizes/rejects any active pending
  proposal, clears `active_proposal` and `active_edit_target` from the restored
  checkpoint, and returns to topology planning; it never retains or rebases an
  old proposal base id. Add accept-vs-revert race tests on SQLite and
  PostgreSQL proving serialization yields one terminal proposal outcome and no
  executable dangling reference. Its operation fence and proposal
  terminalization occur in the same session-write transaction as the restored
  checkpoint metadata.
- [ ] Replace remaining callers with `guided/planning.py`, then delete active
  `ChainProposal`, `PROPOSE_CHAIN`, `solve_chain`,
  `handle_step_3_chain_accept`, `step_3_edit_index`, fixed `chain_in` / `main`
  construction, and their chain-only tests.
- [ ] Add an architecture test scanning active Python/TypeScript code and API
  schema for those removed contracts. Ordinary engine uses of “chain” are not
  forbidden.
- [ ] Add import-boundary/call-count tests proving freeform, guided-full,
  guided-staged, and tutorial each invoke the shared planner and lock-assuming
  commit dependency exactly once. Treat the string scan as a deletion smoke
  test, not the primary architecture proof.

Run:

```bash
uv run pytest \
  tests/integration/web/composer/guided/test_schema8_fork_revert.py \
  tests/unit/web/sessions/test_fork.py \
  tests/unit/web/composer/guided/test_no_chain_authoring_path.py -q
uv run pytest \
  tests/unit/web/composer/guided \
  tests/integration/web/composer/guided -q
cd src/elspeth/web/frontend
npm test -- --run \
  src/components/chat/guided \
  src/stores/sessionStore.guided.test.ts
npm run typecheck
cd ../../../..
uv run ruff check src/elspeth/web/composer/guided src/elspeth/web/sessions
uv run mypy src/elspeth/web/composer/guided src/elspeth/web/sessions
git diff --check
```

Expected: all commands exit 0.

**Definition of done:** Only guided schema 8/session epoch 29 is current; the
backend and frontend speak one `PROPOSE_PIPELINE` protocol backed by the
existing proposal store; fork/revert cannot create cross-session proposal
references; and the transform-only authoring path is unreachable.
