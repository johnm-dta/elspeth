# Composer Capability Parity Plan 03: Guided State and Protocol Replacement

> **For implementers:** Use `superpowers:subagent-driven-development` or
> `superpowers:executing-plans`; apply test-driven development to each task.

**Goal:** Replace the tutorial-era linear guided checkpoint, proposal turn, and
frontend renderer with one strict current-schema model for complete canonical
pipeline proposals.

**Architecture:** This is a pre-release schema break. Guided schema 8 and session
epoch 28 are the only supported current state. Existing pre-release session data
is recreated. There is no version dispatcher, legacy decoder, old request arm,
or runtime architecture setting.

**Depends on:** Plans 01 and 02.

## File structure

**Create:**

- `src/elspeth/web/composer/guided/proposal_audit.py`
- `src/elspeth/web/composer/guided/planning.py` — minimum schema-8 adapter from reviewed facts to the shared planner and commit seams.
- `src/elspeth/web/frontend/src/components/chat/guided/ProposePipelineTurn.tsx`
- `src/elspeth/web/frontend/src/components/chat/guided/ProposePipelineTurn.test.tsx`
- `scripts/recreate-staging-session-state.sh`
- `tests/unit/web/composer/guided/test_current_schema.py`
- `tests/unit/web/composer/guided/test_no_linear_authoring_path.py`
- `tests/unit/deploy/test_recreate_staging_session_state.py`
- `tests/unit/docs/test_staging_session_recreation_policy.py`
- `tests/integration/web/composer/guided/test_proposal_audit_projection.py`
- `tests/integration/web/composer/guided/test_schema_epoch_replacement.py`
- `tests/integration/web/composer/guided/test_schema8_fork.py`

**Modify:**

- `src/elspeth/web/composer/guided/state_machine.py`
- `src/elspeth/web/composer/guided/protocol.py`
- `src/elspeth/web/composer/guided/audit.py`
- `src/elspeth/web/composer/guided/emitters.py`
- `src/elspeth/web/composer/guided/steps.py`
- `src/elspeth/web/sessions/models.py`
- `src/elspeth/web/sessions/converters.py`
- `src/elspeth/web/sessions/routes/_helpers.py`
- `src/elspeth/web/sessions/routes/composer/guided.py`
- `src/elspeth/web/sessions/routes/sessions.py`
- `src/elspeth/web/frontend/src/types/guided.ts`
- `src/elspeth/web/frontend/src/components/chat/guided/GuidedTurn.tsx`
- `src/elspeth/web/frontend/src/stores/sessionStore.ts`
- `docs/runbooks/staging-session-db-recreation.md`

**Delete after callers are replaced:**

- `src/elspeth/web/composer/guided/chain_solver.py`
- `src/elspeth/web/sessions/_guided_solve_chain.py`
- linear-chain payload/model code and tests whose only contract is
  `PROPOSE_CHAIN`, `ChainProposal`, `step_3_edit_index`, `chain_in`, or `main`.

## Task 1: Define the only current guided checkpoint

- [ ] Write failing round-trip and strictness tests for schema 8.
- [ ] Replace singular source/output and transform-chain fields with:

```python
GUIDED_SESSION_SCHEMA_VERSION = 8

reviewed_sources: Mapping[str, SourceResolved]
reviewed_outputs: Mapping[str, SinkOutputResolved]
pending_source_intents: Mapping[str, SourceIntent]
pending_output_intents: Mapping[str, SinkIntent]
active_edit_target: ComponentTarget | None  # {kind, stable_id}
pipeline_proposal: PipelineProposal | None
deferred_intents: tuple[DeferredStageIntent, ...]
```

- [ ] Keep the exact-key, exact-type, closed-enum, recursive-freeze, and
  non-negative-counter invariants used by current Tier-1 state.
- [ ] Test every guided stage, pending proposal, pending stable-id edit,
  completed state, exited state, and process restart.
- [ ] Assert `schema_version != 8` fails closed; do not add a converter.

Run:

```bash
uv run pytest tests/unit/web/composer/guided/test_current_schema.py -q
```

## Task 2: Make the pre-release boundary explicit

- [ ] Write a failing test that opens an epoch-27 SQLite session database and
  expects the existing actionable stale-schema startup error.
- [ ] Bump `SESSION_SCHEMA_EPOCH` from 27 to 28 and extend its history comment:
  schema 8 replaces transform-only guided checkpoint/protocol state.
- [ ] Assert a fresh database receives epoch 28 and accepts schema-8 sessions.
- [ ] Replace the active current runbook procedure for both supported deployment shapes:
  stop the service, back up only if useful for diagnosis, delete/recreate the
  SQLite session DB or drop/recreate the pre-release PostgreSQL session data,
  restart, verify readiness, verify an empty session list, and start a fresh
  guided session.
- [ ] Remove instructions to record/switch to an old source ref, restore an old
  session database, serve archived secrets/state, or otherwise make pre-cutover
  code/data a supported recovery point. A diagnostic archive is non-restorable
  evidence and must be access-controlled and destroyed under the runbook's
  secret-retention policy.
- [ ] Define failure handling as: keep the service drained, preserve sanitized
  diagnostics, fix the current implementation, recreate fresh state, deploy the
  corrected revision, and rerun verification.
- [ ] State that this procedure invalidates resumable composer sessions and
  first-run tutorial progress by design.
- [ ] Add `tests/unit/docs/test_staging_session_recreation_policy.py`; fail if
  the active runbook contains an old-source switch, database restore command, or
  an operational downgrade/recovery section.
- [ ] Implement `scripts/recreate-staging-session-state.sh` with host/service,
  clean-tree, expected-revision, resolved-path, explicit-confirmation,
  open-file, and epoch checks. It may create a short-lived access-controlled
  diagnostic archive, but it has no source-switch or restore arm. On failure it
  leaves staging drained for a corrected build and another fresh recreation.
- [ ] Mark the script executable in git and assert `test -x
  scripts/recreate-staging-session-state.sh` in its unit/plan gate.
- [ ] Unit-test the script with stubbed `systemctl`, `sqlite3`, `curl`, and
  filesystem commands. Wrong host/path/revision/confirmation must fail before
  deletion; post-start failure must never invoke a restore action.

Run:

```bash
uv run pytest tests/integration/web/composer/guided/test_schema_epoch_replacement.py tests/unit/web/sessions/test_schema.py tests/unit/docs/test_staging_session_recreation_policy.py tests/unit/deploy/test_recreate_staging_session_state.py -q
```

## Task 3: Replace the proposal wire contract

- [ ] Add `TurnType.PROPOSE_PIPELINE` with a closed payload containing the
  redacted canonical graph summary, `why`, `draft_hash`, blockers, and stable
  edit targets.
- [ ] Add accept/reject/revise response shapes that must echo `draft_hash`.
- [ ] Remove `TurnType.PROPOSE_CHAIN`, `ProposeChainPayload`, chain response
  parsing, index-based edits, and the corresponding legal-turn entries.
- [ ] Update payload validation, nested-shape maps, response summaries, audit
  allowlists, and API schemas together.
- [ ] Add exhaustive enum/match tests so a new turn cannot be omitted from
  validation, redaction, rendering, or response parsing.

Run:

```bash
uv run pytest tests/unit/web/composer/guided/test_protocol.py tests/integration/web/composer/guided/test_wire_dispatch.py -q
```

## Task 4: Implement custody-safe proposal persistence and audit projection

- [ ] Store the exact deep-frozen, custody-safe `PipelineProposal` only in the
  private resumable checkpoint.
- [ ] Store a separately allowlisted audit projection bound by
  `audit_payload_hash` and `draft_hash`; never execute that projection.
- [ ] Add canaries for credential literals, resolved secrets, raw inline blob
  content, raw provider errors, and raw validation messages.
- [ ] Cover proposed, revised, rejected, validation-failed, accepted, restart,
  hash-conflict, revert, and fork cases under schema 8.
- [ ] In `routes/sessions.py`, remap every proposal `blob_id` when forking a
  session. Abort the fork atomically if any referenced blob cannot be copied and
  rebound to the child; never leave child `composer_meta` pointing at a
  parent-owned blob.
- [ ] Preserve the compose lock and reject stale base-content, anchor, or draft
  hashes with 409 before any state mutation.

Run:

```bash
uv run pytest tests/integration/web/composer/guided/test_proposal_audit_projection.py tests/integration/web/composer/guided/test_audit_emission.py -q
uv run pytest tests/integration/web/composer/guided/test_schema8_fork.py -q
```

## Task 5: Replace the frontend renderer and client state

- [ ] Add `ProposePipelineTurn` using the existing full-DAG and wire-review
  components for graph, node, output, route, and field-contract summaries.
- [ ] Send stable component targets and echoed `draft_hash` for all proposal
  responses.
- [ ] Remove the `ProposeChainTurn` renderer, its type arm, fixtures, and tests.
- [ ] Make TypeScript unions closed over the current server protocol and add a
  compile-time exhaustiveness assertion.
- [ ] Test arbitrary forks, fan-in, multiple sources, multiple outputs, gates,
  queues, aggregations, error paths, revise, reject, and stale-hash display.

Run:

```bash
cd src/elspeth/web/frontend
npm test -- src/components/chat/guided/ProposePipelineTurn.test.tsx src/components/chat/guided/GuidedTurn.test.tsx src/stores/sessionStore.guided.test.ts
npm run typecheck
```

## Task 6: Delete the linear proposal implementation

- [ ] Replace every remaining `solve_chain` and chain-accept caller with
  the minimum `guided/planning.py` adapter, which calls `plan_pipeline()` plus
  `commit_pipeline_proposal()` using schema-8 reviewed facts and the current
  staged prompt. The guided routes and frontend must remain green at the end of
  this task; Plan 04 generalizes this adapter for guided-full/tutorial, plural
  interaction, and stage handling, not basic compilation or proposal dispatch.
- [ ] Delete chain-only models, emitters, helpers, renderers, tests, and prompt
  claims.
- [ ] Add an architecture test that fails on active references to:
  `ChainProposal`, `ProposeChainPayload`, `PROPOSE_CHAIN`, `solve_chain`,
  `handle_step_3_chain_accept`, or index-based transform edits.
- [ ] Keep ordinary words such as “chain” only where they describe engine graph
  behavior, not the removed guided authoring contract.

Run:

```bash
uv run pytest tests/unit/web/composer/guided/test_no_linear_authoring_path.py tests/integration/web/composer/guided -q
```

## Plan 03 completion gate

```bash
uv run pytest \
  tests/unit/web/composer/guided/test_current_schema.py \
  tests/unit/web/composer/guided/test_no_linear_authoring_path.py \
  tests/integration/web/composer/guided/test_schema_epoch_replacement.py \
  tests/integration/web/composer/guided/test_proposal_audit_projection.py \
  tests/integration/web/composer/guided/test_schema8_fork.py \
  tests/integration/web/composer/guided/test_wire_dispatch.py -q
cd src/elspeth/web/frontend
npm test -- src/components/chat/guided src/stores/sessionStore.guided.test.ts
npm run typecheck
cd ../../../..
uv run ruff check src/elspeth/web/composer/guided src/elspeth/web/sessions
git diff --check
```

Plan 03 passes only when the repository has one guided schema and proposal
protocol, stale pre-release state fails with the documented recreation action,
and no transform-only authoring path remains.
