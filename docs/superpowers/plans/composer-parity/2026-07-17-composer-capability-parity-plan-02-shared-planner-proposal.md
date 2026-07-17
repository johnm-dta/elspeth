# Composer Capability Parity Plan 02: Shared Planner and Proposal Lifecycle

**Goal:** Add one custody-safe canonical planner and commit adapter that every
authoring surface can use without introducing a second proposal store or DAG
schema.

**Architecture:** `PipelineProposal` is an immutable server-side envelope over
exact `set_pipeline` arguments. `plan_pipeline()` performs read-only discovery,
strict terminal parsing, side-effect-free candidate validation, custody
finalization, custody-safe revalidation, and bounded repair. The existing
`composition_proposals` service persists the valid draft;
the existing acceptance route remains the transaction and cancellation-safety
authority.

**Prerequisite:** Plan 01 passes. This phase does not bump the session schema.

**Lock and authority contract:** The request route acquires the shared
per-session compose lock first. While holding it, helpers may perform a
filesystem custody write outside a database transaction and open short
blob/session write transactions; no code may acquire the compose lock while a
database transaction is open. Filesystem custody is idempotent and recoverable,
not transactionally atomic with either database. Proposal creation remains
split from the existing compose/LLM audit store: finish the audit record first,
then create the proposal row plus `proposal.created` event in one session-store
transaction. Only that row and event authorize reconstruction; an audit record
left behind when proposal creation fails is diagnostic and never reviewable.

## Task 1: Define hashes and the envelope without a topology model

**Files:**

- Create: `src/elspeth/web/composer/pipeline_proposal.py`
- Create: `tests/unit/web/composer/test_pipeline_proposal.py`
- Modify: `src/elspeth/web/sessions/routes/composer/guided.py`

- [ ] Write failing tests for recursive immutability, deterministic hashes,
  safe `secret_ref` preservation, and rejection of a supplied hash mismatch.
- [ ] Move the existing `_composition_content_hash()` implementation from
  `sessions/routes/composer/guided.py` into the new module without changing its
  preimage.
- [ ] Implement these contracts using existing `canonical_json()`,
  `stable_hash()`, `deep_freeze()`, and `freeze_fields()` utilities:

```python
class PlannerSurface(StrEnum):
    FREEFORM = "freeform"
    GUIDED_FULL = "guided_full"
    GUIDED_STAGED = "guided_staged"
    TUTORIAL_PROFILE = "tutorial_profile"


@dataclass(frozen=True, slots=True)
class AbsentBase:
    kind: Literal["absent"] = "absent"


@dataclass(frozen=True, slots=True)
class PresentBase:
    state_id: UUID
    composition_content_hash: str
    kind: Literal["present"] = "present"


ProposalBase = AbsentBase | PresentBase


@dataclass(frozen=True, slots=True)
class PipelineProposal:
    pipeline: Mapping[str, Any]
    draft_hash: str
    base: ProposalBase
    reviewed_anchor_hash: str
    surface: PlannerSurface
    repair_count: int
    skill_hash: str
    covered_deferred_intent_ids: tuple[str, ...] = ()
    supersedes_draft_hash: str | None = None
```

`pipeline` contains the exact canonical arguments and no duplicate node, edge,
source, or output dataclasses. Compute `draft_hash` over a domain-separated
versioned object, including the tagged base and ordered covered-intent ids.
`AbsentBase` means that no current composition state exists; `PresentBase`
always binds both the exact state id and its composition-content hash. There is
no nullable or wildcard base. Compute the reviewed-anchor hash over deep-frozen
reviewed facts. Reject negative repair counts, duplicate or malformed intent
ids, and any hash mismatch on construction or restore. Do not persist model-authored
rationale: generate public summary and rationale with the existing server-owned
`build_tool_proposal_summary()` over redacted canonical arguments. If model
rationale is useful for diagnostics, retain only its domain-separated hash.

Run:

```bash
uv run pytest tests/unit/web/composer/test_pipeline_proposal.py -q
```

Expected: PASS.

## Task 2: Finalize inline content into existing blob custody

**Files:**

- Create: `src/elspeth/web/composer/pipeline_custody.py`
- Modify: `src/elspeth/contracts/blobs.py`
- Modify: `src/elspeth/web/blobs/service.py`
- Modify: `src/elspeth/web/composer/audit.py`
- Modify: `src/elspeth/web/composer/tools/blobs.py`
- Modify: `src/elspeth/web/composer/tools/sessions.py`
- Create: `tests/integration/web/composer/test_pipeline_custody.py`
- Modify: `tests/unit/web/blobs/test_service.py`

- [ ] Write failing tests submitting `source.inline_blob.content` and asserting
  that the reviewable arguments contain `source.blob_id` instead, with no raw
  content in proposal rows, event payloads, LLM/tool audit, validation errors,
  logs, or API responses.
- [ ] Add retry and interruption cases after reservation, file write, blob-row
  finalization, and before proposal creation. The same session, user message,
  MIME metadata, and content hash must reuse one blob; any mismatch fails
  closed.
- [ ] Derive a stable, domain-separated custody key from session id,
  originating message id, content hash, MIME type, filename, and creation
  provenance. Hold the same-session write lock (and the repository's equivalent
  PostgreSQL row/advisory serialization) across lookup/reservation/finalization;
  validate every field before reuse. Test concurrent callers on SQLite and
  PostgreSQL, orphan recovery after file write, mismatch rejection, and quota
  charged exactly once.
- [ ] Derive the blob primary key as a domain-separated UUID5 (128-bit,
  UUID-column-compatible) from that custody key; never store an arbitrary hash
  in `blobs.id`. Rely on existing primary-key uniqueness. On a concurrent
  insert conflict, load and validate the winning row byte-for-byte against the
  reservation before reuse; a mismatch is an integrity error. Do not add a
  second uniqueness mechanism or a Plan-02 schema bump.
- [ ] Add an idempotent `reserve_inline_custody()` operation to the existing
  blob service by extracting and reusing the storage/provenance/quota primitive
  currently shared conceptually by `_prepare_blob_create()`,
  `_persist_prepared_blob_create()`, and `BlobServiceImpl.create_blob()`. The
  old executor and the new proposal path delegate to that primitive; do not
  leave three independent custody implementations. Keep its row, registry,
  path confinement, provenance, retention, and cleanup rules.
- [ ] Implement `finalize_pipeline_custody()` to materialize only the legacy
  `source.inline_blob` shape currently accepted by canonical `set_pipeline`,
  remove the entire `source.inline_blob` member, set `source.blob_id`, and
  compute the proposal hash afterward. Never leave both forms, write
  `source.options.blob_ref`, or claim named-source inline blob support while the
  canonical validator rejects it.
- [ ] Route every reviewable `set_pipeline` proposal through this function,
  including terminal planner output and an ordinary freeform compose-loop tool
  call intercepted in explicit-approval mode on a non-empty composition.
  Centralize this before the final candidate build, hash, and proposal creation
  so all three observe the same `blob_id` arguments and no caller can persist
  raw inline content. Add that non-empty explicit-approval regression.
- [ ] On proposal rejection, leave the materialized blob under current session
  retention rather than adding destructive cleanup to the approval path. A
  blob referenced by a pending proposal is retained; after terminal rejection
  or supersession, an unreferenced custody blob follows the existing
  session/blob retention and reconciliation policy. Do not add a permanent pin
  or a new GC ledger.

Run:

```bash
uv run pytest \
  tests/integration/web/composer/test_pipeline_custody.py \
  tests/integration/web/composer/test_inline_source_provenance.py \
  tests/unit/web/blobs/test_service.py -q
```

Expected: PASS; retries create one ready blob and persisted proposal/audit
surfaces contain no raw inline bytes.

## Task 3: Implement one read-only planner loop

**Files:**

- Create: `src/elspeth/web/composer/pipeline_planner.py`
- Create: `tests/unit/web/composer/test_pipeline_planner.py`
- Create: `src/elspeth/web/composer/tools/schema_contract.py`
- Create: `tests/unit/web/composer/test_tool_schema_contract.py`
- Modify: `src/elspeth/web/composer/audit.py`
- Reference: `src/elspeth/web/composer/tools/sessions.py`

- [ ] Write failing tests that send canonical fixtures through a deterministic
  fake completion using the real terminal tool-call parser. Do not inject a
  preconstructed proposal or state.
- [ ] Expose a defensive-copy accessor in `tools/schema_contract.py` by
  selecting `set_pipeline` from `get_tool_definitions()`. Do not reach into the
  private `_TOOL_DEFS_BY_NAME` registry. Add a directional compatibility test
  against `SetPipelineArgumentsModel.model_json_schema()`: every typed-valid
  canonical shape must be advertizable, requiredness/nullability must not
  conflict, and all supported source/node/edge/output shapes must be reachable.
  Do not require byte equality or identical prose/strictness; the LLM schema and
  Pydantic security boundary are intentionally distinct artifacts, and runtime
  validation may be stricter about extra properties.
- [ ] Implement `planner_terminal_tool_definition()` with a `pipeline` property
  equal to that registered declaration. Do not ask the model for a public
  rationale field; public summary/rationale are generated by the server after
  redaction.
- [ ] Implement `plan_pipeline()` with inputs for intent, current state,
  reviewed facts, surface, request-scoped policy catalog/plugin snapshot,
  originating message, a server-constructed `ProposalBase`, model
  configuration, rendered skill, and repair budget. The planner seals that base
  into the immutable envelope exactly once; callers may not replace it after
  the draft hash is computed.
- [ ] Advertise the current read-only composer discovery tools plus the terminal
  proposal tool. Reject mutation calls and unknown discovery tools.
- [ ] Reuse the current discovery/composition turn caps, provider deadline,
  prompt/token limits, anti-repeat/anti-anchor controls, request rate limiter,
  in-flight accounting, progress settlement, and client-disconnect cancellation
  used by ordinary compose/guided routes. Bound total provider calls as well as
  repairs. Test 429, deadline, disconnect, repeated discovery, malformed
  response, prompt/cost budget exhaustion, and zero residual state/proposal
  mutation.
- [ ] Strictly parse the terminal payload and build a preliminary,
  side-effect-free candidate using prepared inline content. If it is invalid,
  feed only the existing allowlisted structured validation projection back for
  bounded repair and create no ready blob. Only after a draft is otherwise
  acceptable, finalize custody, replace inline content with the resulting blob
  ids, rebuild/revalidate the custody-safe candidate, hash it, and return the
  `PipelineProposal`. Failed drafts create no proposal row and publish no state.
- [ ] Record hashes of the actual rendered messages and advertised tool schemas
  using the existing LLM audit path. Do not copy raw provider errors into repair
  feedback or audit.

Run:

```bash
uv run pytest \
  tests/unit/web/composer/test_pipeline_planner.py \
  tests/unit/web/composer/test_tool_schema_contract.py \
  tests/unit/web/composer/test_audit.py -q
```

Expected: PASS; one repair may replace an invalid draft, an exhausted budget
fails closed, and no planner discovery call mutates state.

## Task 4: Adapt the existing proposal lifecycle and acceptance route

**Files:**

- Create: `src/elspeth/web/composer/pipeline_commit.py`
- Modify: `src/elspeth/web/sessions/service.py`
- Modify: `src/elspeth/web/sessions/protocol.py`
- Modify: `src/elspeth/web/sessions/models.py`
- Modify: `src/elspeth/web/sessions/routes/composer/proposals.py`
- Modify: `src/elspeth/web/sessions/routes/_helpers.py`
- Modify: `src/elspeth/web/sessions/schemas.py`
- Modify: `src/elspeth/web/frontend/src/types/index.ts`
- Modify: `src/elspeth/web/frontend/src/api/client.ts`
- Modify: `src/elspeth/web/frontend/src/stores/sessionStore.ts`
- Modify: `src/elspeth/web/frontend/src/stores/sessionStore.test.ts`
- Create: `tests/integration/web/composer/test_pipeline_proposal_lifecycle.py`
- Modify: `tests/unit/web/sessions/test_composer_proposals.py`

- [ ] Write failing tests proving a valid planner draft creates exactly one
  existing `composition_proposals` row with exact custody-safe arguments,
  redacted public arguments, current provenance, and base state id.
- [ ] Add the surface, draft/base/anchor/audit hashes and repair count to a
  closed, explicitly versioned `pipeline_proposal_created.v1` payload on the
  allowlisted `proposal.created` event. Extend
  the Python and TypeScript public response with optional safe fields so legacy
  non-pipeline proposals remain representable. Do not add SQL columns for values
  recomputable from exact arguments or already anchored in the state/checkpoint.
- [ ] Version the event payload explicitly. During Plan 02 readers accept the
  exact legacy creation shape or the new pipeline-metadata shape; pipeline
  reconstruction requires the latter. After the epoch-29 recreation, every new
  canonical pipeline proposal must carry the new shape. Do not silently reinterpret
  an old event as current metadata.
- [ ] Implement a service helper that reconstructs and verifies a
  `PipelineProposal` from the private row and its authoritative immutable
  `proposal.created` event. Load the proposal once and query exactly once by
  `(session_id, proposal_id, event_type="proposal.created")`; require exactly
  one event and never use the proposal row's mutable `audit_event_id`, which is
  replaced by later lifecycle events. Validate the closed versioned metadata
  shape, recompute draft/base/audit hashes, and compare the stored anchor to
  current server-loaded reviewed facts. Echoed public hashes are untrusted
  concurrency tokens, never restore authority. Cross-session ids, wrong tool
  names, altered exact arguments, missing/duplicate events, and hash mismatches
  raise an integrity error before dispatch.
- [ ] Add an `AcceptProposalRequest` hash echo for pipeline proposals. The HTTP
  route requires `draft_hash` when the creation event marks a pipeline proposal;
  legacy non-pipeline proposals keep their current no-hash behavior. Update the
  frontend client/store and test stale/missing/mismatched echoes.
- [ ] Split `accept_composition_proposal()` into a lock-assuming
  `prepare_pipeline_proposal_commit()` and a persistence settlement. The request
  route remains the only compose-lock owner; no shared helper reacquires the
  non-reentrant lock. Preparation preserves ownership, base-state conflict,
  current policy catalog/snapshot, exact replay, candidate revalidation,
  deferred cancellation, and 409/422 semantics, but performs no state/proposal
  database settlement.
- [ ] Enforce the tagged base under that same lock and again inside settlement.
  `AbsentBase` is valid only while no current composition state exists;
  creation of a first state before settlement returns 409. `PresentBase`
  requires both the current state id and composition-content hash to match;
  the settlement transaction reloads and compares both, never id alone.
  Add absent-vs-first-state, same-content/new-state, and concurrent-state-change
  regressions; never treat a missing state id as “accept against anything.”
- [ ] Add one `SessionServiceProtocol` method that, in one session-write
  transaction, calls internal `_insert_composition_state()`, appends the
  terminal proposal event, updates proposal status/committed state id, and
  writes caller-supplied final guided composer metadata when present. Acceptance
  must call only this protocol method; it must not call the public state-save API
  in a separate transaction. The method validates that the proposal is still
  pending and its base id/content hash are still current. A duplicate retry
  returns the already-bound outcome only when every id/hash matches; otherwise
  it fails closed.
- [ ] Before publishing, compare candidate content hash with the executor result
  content hash. A mismatch settles the proposal as failed/rejected with an
  allowlisted reason and publishes no new current state.
- [ ] Every reject/fail/supersede settlement records a closed allowlisted reason
  code in its terminal event. Do not persist raw validation, provider, or
  exception text as the reason.
- [ ] Add interruption/retry cases before/after dispatch and before/after the
  atomic settlement. Assert there is never a published state with a pending
  proposal, a committed proposal with stale guided metadata, or more than one
  committed state/terminal event. A crash after proposal staging but before
  acceptance may leave one valid pending proposal and is recoverable.
- [ ] Keep the generic proposal endpoint for legacy proposals and
  `GUIDED_FULL`, but reject a canonical proposal whose authoritative creation
  metadata says `GUIDED_STAGED` or `TUTORIAL_PROFILE`. Those surfaces must use
  the guided transition so proposal settlement, covered-intent consumption,
  and checkpoint advancement cannot be bypassed.
- [ ] Emit low-cardinality planner, custody, commit, and integrity-conflict
  counters only after the corresponding audit write. Use closed surface/result
  enums and never attach hashes, ids, user text, filenames, or exception text.

Run:

```bash
uv run pytest \
  tests/integration/web/composer/test_pipeline_proposal_lifecycle.py \
  tests/unit/web/sessions/test_composer_proposals.py \
  tests/unit/web/sessions/test_routes.py -q
```

Expected: PASS; HTTP and internal acceptance call the same commit adapter and
accepted exact arguments equal the arguments seen at the audited executor
boundary.

## Task 5: Route the freeform new-pipeline tracer through the planner

**Files:**

- Modify: `src/elspeth/web/composer/service.py`
- Modify: `src/elspeth/web/composer/protocol.py`
- Modify: `src/elspeth/web/composer/tool_batch.py`
- Modify: `src/elspeth/web/sessions/routes/messages.py`
- Modify: `src/elspeth/web/sessions/routes/composer/compose.py`
- Create: `tests/integration/web/composer/test_freeform_pipeline_planner.py`

- [ ] Add a failing `ComposerServiceImpl.compose()` test for an empty
  composition and an end-to-end build request. It must traverse request prompt
  assembly, `plan_pipeline()`, candidate validation, and the existing trust-mode
  behavior.
- [ ] Define new/empty by canonical topology content (no sources, nodes, edges,
  or outputs), not state version or composer metadata. Route only that
  full-pipeline authoring through `plan_pipeline()`.
  When `_try_apply_freeform_recipe_intent()` produces a full-pipeline result for
  an empty/new composition, normalize it into the same durable proposal and
  `PipelineCommitIntent`; it may not save state directly. Preserve the ordinary
  compose loop only for established incremental edits, including the current
  profile-aware transform splice path.
- [ ] Add a `PipelineCommitIntent` to `ComposerResult`. In `auto_commit`, the
  service creates/stages the valid durable proposal and returns a commit intent;
  it does not persist composition state or reacquire the route lock. Both
  `sessions/routes/messages.py` and `sessions/routes/composer/compose.py` pass
  that intent through the lock-assuming preparation and atomic settlement
  instead of their normal state-save arm. In `explicit_approve`, return the
  existing pending proposal response without publishing state.
- [ ] Keep the current generic acceptance behavior for non-pipeline mutation
  proposals. Select the pipeline coordinator only when the authoritative
  creation event identifies a canonical pipeline proposal.
- [ ] Prove both trust modes share planner, candidate, proposal, and commit
  code; neither duplicates the `set_pipeline` executor or topology schema.

Run:

```bash
uv run pytest \
  tests/integration/web/composer/test_freeform_pipeline_planner.py \
  tests/integration/web/composer/test_freeform_proposal_prevalidation.py \
  tests/unit/web/composer/test_service.py -q
uv run ruff check src/elspeth/web/composer src/elspeth/web/sessions tests/integration/web/composer
uv run mypy src/elspeth/web/composer src/elspeth/web/sessions
git diff --check
```

Expected: all commands exit 0.

**Definition of done:** A production freeform new-pipeline request can derive,
review, and commit a full graph through one custody-safe canonical proposal
lifecycle. Existing incremental editing and proposal APIs remain compatible,
and this phase adds no new persisted schema version.
