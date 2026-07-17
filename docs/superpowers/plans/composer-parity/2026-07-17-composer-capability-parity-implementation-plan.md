# Composer Capability Parity Implementation Plan Set

**Status:** Current implementation plan
**Baseline:** `release/0.7.1` at `cc593f3a7ae29cc52d94bd82661fbdfb04e5fd81`
**Controlling issue:** `elspeth-7e2dd67275`
**Design:** [Composer Guided/Freeform Capability Parity](2026-07-13-composer-guided-freeform-capability-parity-design.md)
**Review:** Approved — [substantive plan review](2026-07-17-composer-capability-parity-implementation-plan.md.review.json)

**Goal:** Let freeform, guided-full, and guided-staged author the same valid
canonical `set_pipeline` graphs while preserving the current security,
reliability, policy, proposal-audit, and profile-validation guarantees.

**Architecture:** Keep `SetPipelineArgumentsModel` and the registered
`set_pipeline` declaration as the only topology language. Extract a
side-effect-free candidate boundary from the current executor, add one shared
planner that returns canonical arguments, and extend the existing
`composition_proposals` lifecycle for review and acceptance. Guided mode stores
reviewed interaction facts, deferred intent, and a reference to the durable
proposal; it does not gain another DAG model.

**Tech stack:** Python 3.13, FastAPI, Pydantic, SQLAlchemy, SQLite/PostgreSQL,
React/TypeScript, Vitest, Playwright, pytest, Hypothesis, LiteLLM.

## Current ground truth

- `SetPipelineArgumentsModel` already accepts the full web-authorable graph:
  legacy or named sources, transforms, gates, aggregations, queues, coalesces,
  explicit edges, and named outputs.
- `_execute_set_pipeline()` already owns canonical construction, current plugin
  policy, operator-profile validation, interpretation-review reconciliation,
  path confinement, credential wiring, and inline-blob preparation.
- `composition_proposals` already stores exact private arguments, a redacted
  public projection, model/skill/argument provenance, base and committed state
  ids, and lifecycle events. Acceptance already runs under the compose lock,
  rejects a stale base state, replays through `execute_tool()`, defers
  cancellation until settlement, and commits a new immutable state.
- Freeform explicit-approval interception currently persists a proposal before
  semantic `set_pipeline` validation. Invalid proposals therefore bypass the
  normal repair loop until a person tries to accept them.
- Guided still persists `ChainProposal`, emits `PROPOSE_CHAIN`, asks a
  transform-only solver for `steps`, and materializes fixed `chain_in` / `main`
  wiring in `handle_step_3_chain_accept()`.
- Current schema constants are `GUIDED_SESSION_SCHEMA_VERSION = 7`,
  `SESSION_SCHEMA_EPOCH = 28`, and `SQLITE_SCHEMA_EPOCH = 27`. The guided
  replacement therefore uses guided schema 8 and session epoch 29. It does not
  change the landscape epoch.
- The repository has live and tutorial workflow profiles; it has no deployed
  `guided_full` endpoint. The endpoint is implementation work, not an assumed
  existing surface.

## Decisions that replace the retired plans

1. Extend the existing proposal row and lifecycle; do not add a parallel
   proposal table, receipt sidecar, sink-effect operation-parent ledger, or
   plan manifest. The epoch-29 guided-operation table is limited to durable
   HTTP retry identity for the two new guided POST surfaces.
2. Land a freeform validate-before-review slice first, without changing a
   persisted schema. This proves the candidate seam against production behavior
   before guided state depends on it.
3. Keep proposal metadata that is needed only for verification in the existing
   event payload and guided checkpoint. Do not add SQL columns merely to mirror
   the in-memory `PipelineProposal` envelope.
4. Materialize inline source content into the existing session-blob store before
   a planner proposal becomes reviewable. Exact proposals may keep safe
   `secret_ref` markers but never raw inline content, credential literals, or
   resolved secrets.
5. Replace guided schema, protocol, backend, and frontend in one session-epoch
   boundary. Reject old stores using the existing pre-release recreate policy;
   do not build a v7 decoder, migration, dual write, feature flag, or downgrade
   path.
6. A fork copies reviewed guided facts but invalidates any pending proposal and
   stable edit target in the child. The child replans against its copied base
   state. It never points at a proposal or blob owned by the parent session.
7. Tutorial remains a teaching profile over the ordinary staged planner. It may
   constrain the lesson and visible controls, not the canonical schema,
   discovery set, or commit path.
8. Coalesce runtime recovery, empty-output evidence, public typed LLM query
   configuration, branch signing, and release packaging are outside this
   feature unless implementation demonstrates that one is necessary for a
   parity acceptance case.

## Dependency order

1. [Plan 01 — Candidate and approval seam](2026-07-17-composer-capability-parity-plan-01-candidate-approval-seam.md)
   extracts and characterizes canonical candidate construction, then validates
   freeform `set_pipeline` proposals before persistence.
2. [Plan 02 — Shared planner and proposal lifecycle](2026-07-17-composer-capability-parity-plan-02-shared-planner-proposal.md)
   adds custody-safe `PipelineProposal`, one planner, and one commit adapter over
   the existing proposal service and acceptance route.
3. [Plan 03 — Guided schema and protocol replacement](2026-07-17-composer-capability-parity-plan-03-guided-replacement.md)
   performs the schema-8/session-29 cutover and deletes the transform-only
   protocol atomically.
4. [Plan 04 — Guided authoring and shared capability](2026-07-17-composer-capability-parity-plan-04-guided-authoring.md)
   adds plural reviewed facts, stage deferral/back-edit, guided-full, staged and
   tutorial controllers, and one shared capability prompt core.
5. [Plan 05 — Parity proof and staging acceptance](2026-07-17-composer-capability-parity-plan-05-verification-acceptance.md)
   builds the real-path corpus, generated-DAG checks, frontend coverage, store
   recreation proof, and the two-LLM colour acceptance.

Each plan ends in a working, tested integration boundary. Development commits
may land between boundaries, but staging continues to expose only the current
architecture until all five plans pass.

## Load-bearing invariants

- Every authoring surface terminates in exact canonical `set_pipeline`
  arguments and commits through the same audited executor.
- Candidate construction and proposal review publish no composition state.
- The accepted arguments are revalidated against the current catalog, policy,
  operator profile, and base state while the compose lock is held.
- The request route acquires the compose lock exactly once. Shared planner and
  commit helpers assume it is held; they never reacquire it. After execution,
  one session-service transaction inserts the immutable state, settles the
  proposal event/status, and writes final guided metadata when applicable.
- A stale state id, draft hash, reviewed-anchor hash, cross-session reference,
  or candidate/commit content mismatch fails closed before publication.
- Proposal bases are explicit: absence means no persisted state exists, while a
  present base binds both state id and content hash. Neither form is a wildcard.
- The two new guided POST operations use durable client operation ids. Ordinary
  guided creation enters through `POST /guided/start`; every post-provider
  custody/proposal/result write is fenced by the current lease token and
  attempt so a late worker cannot settle after takeover.
- Proposal and audit APIs expose only the redacted projection. Private replay
  arguments contain no raw inline bytes or resolved credentials.
- Repair feedback uses the existing allowlisted structured validation shape;
  raw exceptions and provider messages are not copied into audit or prompts.
- Wrong-stage intent is retained or rewound by stable id. It is not discarded,
  called unsupported, counted as a repair, or used to advance a stage.
- Removing the old path means deleting active `ChainProposal`,
  `PROPOSE_CHAIN`, `solve_chain`, `handle_step_3_chain_accept`, step-index edit,
  and linear proposal renderer references in the same schema cutover.

## Verification gates

After every plan:

```bash
uv run ruff check src tests
uv run mypy src
uv run pytest -q
cd src/elspeth/web/frontend && npm run typecheck && npm test -- --run
git diff --check
```

Expected: every command exits 0. If the full suite is too slow for the inner
TDD loop, run the focused commands in the phase plan first; the full gate is
still required before moving to the next plan.

Before staging:

```bash
uv run pytest tests/testcontainer/web -q
uv run pytest tests/property -q
cd src/elspeth/web/frontend && npm run build && npm run test:e2e
```

Expected: all suites pass without parity skips or expected failures.

## Completion definition

The controlling issue can close only when:

- freeform, guided-full, and guided-staged each derive and commit all nine
  canonical topology fixtures through production request parsing, planner tool
  parsing, candidate validation, proposal persistence, acceptance, and audited
  dispatch;
- generated valid DAGs are accepted equivalently across those three surfaces;
- tutorial proves schema, discovery, planner, and commit identity while its
  fixed lesson journey remains green;
- stale guided/session data fails with the documented recreate instruction;
- the two-LLM colour graph is independently derived in all three ordinary
  surfaces, runs successfully, produces the exact business fields and output
  counts, and leaves a closed redacted audit trail;
- repository search and architecture tests show no active transform-only guided
  proposal path; and
- current release, security, reliability, and integrity gates pass.
