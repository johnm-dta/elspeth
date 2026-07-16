# Composer Capability Parity — Implementation Prompt

> **RETIRED EXECUTION PACKAGE (2026-07-17): DO NOT COPY OR EXECUTE.** This
> prompt was reviewed against an obsolete release baseline and contains stale
> schema, recovery, and deployment mechanics. See
> [the current disposition](2026-07-17-current-plan-disposition.md).

Copy the prompt below into a fresh implementation agent.

---

You are implementing the complete Composer guided/freeform capability-parity
feature in `/home/john/elspeth` on branch `release/0.7.1`.

This is a high-priority, end-to-end delivery assignment. Do not stop after
writing more plans or making one path pass. Implement all seven slices, deploy
the integrated feature to staging, execute the live three-surface colour proof,
fix every in-scope defect you uncover, and close the controlling Filigree issue
only when the full acceptance gate passes.

## Authoritative package

Read these files before editing code:

1. `docs/superpowers/plans/composer-parity/2026-07-13-composer-guided-freeform-capability-parity-design.md`
2. `docs/superpowers/plans/composer-parity/2026-07-13-composer-capability-parity-implementation-plan.md`
3. `docs/superpowers/plans/composer-parity/2026-07-13-composer-capability-parity-implementation-plan.review.json`
4. Plans 01 through 07 in the same directory, in numeric order.
5. `docs/superpowers/specs/2026-07-13-two-llm-colour-hybrid-pipeline-design.md`
6. `docs/superpowers/plans/2026-07-13-two-llm-colour-hybrid-demo-run-sheet.md`
7. The current repository `AGENTS.md` and every more-specific `AGENTS.md` that
   governs a file you touch.

The approved review report has zero blockers. If live code has changed since the
review, treat the current code and repository instructions as factual authority,
then update the affected plan precisely before implementing it. Do not silently
invent a different architecture.

## Non-negotiable product result

Freeform, guided-full, guided-staged, and tutorial-profile must use one complete
pipeline language, one `plan_pipeline()` implementation, one canonical
`PipelineProposal`, and one audited `set_pipeline` commit seam.

Guided is a staged interaction model, not a reduced-capability composer. It must
author every web-valid canonical pipeline that freeform can author, including
multiple sources and outputs, gates, queues, forks, coalesces, aggregations, row
expansion, structured LLM transforms, explicit edges, and failure routes.

The tutorial is a guided profile. It may change teaching copy and fixed lesson
data only; it may not narrow schemas, tools, catalog assistance, discovery, or
planner capability.

When a guided user requests a valid plugin at the wrong stage:

- if its stage is ahead, say clearly that they need to wait, persist the intent,
  and consume it automatically at the correct stage;
- if its stage has already been reviewed, use stable-id back/edit;
- if the name is ambiguous across plugin kinds, ask for disambiguation;
- if it is unavailable or forbidden, report that distinct catalog/policy error;
- never configure it in the wrong component, discard it, advance silently,
  spend a repair, or claim guided cannot express it.

## Forward-only pre-release policy

ELSPETH is pre-release software. Implement the feature and fix it forward.

- Bump `GUIDED_SESSION_SCHEMA_VERSION` from 7 to 8 and
  `SESSION_SCHEMA_EPOCH` from 27 to 28.
- Delete/recreate incompatible pre-release session state using the guarded
  staging procedure in Plan 03/07.
- Remove the active `ChainProposal`, `PROPOSE_CHAIN`, `solve_chain`, linear
  materializer, index-based edit, backend request, and frontend renderer paths.
- Do not build a version-7 reader, state migration, dual protocol, old/new
  architecture flag, sticky writer, fallback executor, downgrade path, or
  old-source/database restoration procedure.
- On failure, keep staging drained if necessary, diagnose the current feature,
  add a regression, fix it, recreate fresh state, redeploy, and reverify.

Do not create a worktree. Work in the supplied checkout and preserve unrelated
changes. Do not use destructive git operations.

Wardline is not a mandatory completion gate for this activity under the current
operator direction. Do not block implementation or ticket closure on it, and do
not modify or wait for the independent Wardline-baseline issue. Continue to run
the targeted trust-boundary, custody, redaction, and security tests named by the
plans. If live repository instructions materially conflict with this direction,
surface the exact conflict to the operator rather than inventing a workaround.

## Tracker boundaries

The controlling issue is:

- `elspeth-7e2dd67275` — guided chains cannot express gate nodes.

Its old suggested remedy—expanding `ChainProposal`—is superseded. Add a tracker
comment linking the canonical-proposal design before coding, then start work
atomically. Attribute tracker actions to your agent identity:

```bash
AGENT_ID="${AGENT_ID:-codex-composer-parity}"
filigree add-comment elspeth-7e2dd67275 \
  "The approved canonical-proposal design supersedes the old ChainProposal expansion remedy. See docs/superpowers/plans/composer-parity/." \
  --actor "$AGENT_ID"
filigree start-work elspeth-7e2dd67275 \
  --assignee "$AGENT_ID" \
  --actor "$AGENT_ID" \
  --advance \
  --commit "release/0.7.1@$(git rev-parse HEAD)"
```

The related row-union/statistics issue `elspeth-a5b86149d4` is being worked
independently. Do not claim, close, or absorb it into this feature. Use its
landed behavior if available; otherwise preserve the explicit dependency/gap in
evidence instead of duplicating another implementation.

Do not file an observation for any defect required by this feature. Fix it in
scope, expand the controlling issue, or create a durable dependent issue.

## Required working method

Use all relevant skills and subagents. At minimum:

- use `loomweave-workflow` to refresh and query the code map before broad text
  searches;
- use `warpline-workflow` before high-blast-radius edits and to derive each
  slice's re-verification worklist;
- use `filigree-workflow` for atomic issue transitions and evidence comments;
- execute with `superpowers:subagent-driven-development`;
- apply `superpowers:test-driven-development` to every behavior change;
- apply `superpowers:systematic-debugging` to every test, staging, provider,
  browser, or runtime error before editing;
- request independent specification and code-quality review after each green
  task/slice;
- apply `superpowers:verification-before-completion` before every commit and
  before the final report;
- use the `playwright` skill for the deployed guided-staged journey.

Use a fresh implementation subagent per bounded task where practical. Do not
let two agents edit the same files concurrently. The main agent owns integration,
reviews, tracker state, deployment, and final evidence.

## Execution sequence

Execute the master run sheet and Plans 01–07 in order. Do not skip a slice.

For every task:

1. Verify the named paths and symbols against current code.
2. Write the specified failing test and run it to prove the failure.
3. Implement the smallest canonical change.
4. Run the narrow test and the task regression set.
5. Run the slice's static, frontend, security, and architecture guards.
6. Obtain specification-compliance and code-quality reviews.
7. Fix all blocking findings and rerun fresh verification.
8. Commit the green task with only its intended files.
9. Record the commit and exact evidence in the master ledger and Filigree.

Do not expose a partially replaced composer architecture to staging. Plans
01–06 must pass together on one integrated revision before Plan 07 deployment.

The following seams are load-bearing:

- coalesce `on_error` must exist throughout runtime, composer state, and YAML
  import/export before schema lock;
- the registered canonical `set_pipeline` declaration and
  `SetPipelineArgumentsModel` must stay structurally identical;
- secret-reference construction probes must expose exact guaranteed fields;
- inline bytes must enter idempotent blob custody before proposal hashing;
- pre-validation custody must create no public proposal row;
- one validated draft creates exactly one public `composition_proposals` row
  used for review, acceptance, retry, and commit;
- executor validity and candidate/result hash equality must pass before current
  state publication;
- freeform production acceptance, guided-full, `/guided/start`, and tutorial
  must all traverse the shared planner and commit seams;
- schema-8 session fork must remap proposal blobs atomically to the child;
- no active or compatibility linear-chain authoring reference may remain.

## Live acceptance

Use `https://elspeth.foundryside.dev/`. Obtain the staging username and password
from the operator and expose them only through the environment variables named
in Plan 07. Never commit credentials, cookies, provider secrets, authorization
headers, or raw model responses.

Use the exact committed ten-row CSV and exact committed plain-English request.
The request may say “two LLMs,” ask for JSON, and ask to remove fields, but it
must not name composer tools, prescribe tool-call order, provide argument JSON,
or import prepared YAML.

Run three independent proofs on one deployed revision:

1. production freeform big-bang;
2. the authenticated deployed guided-full server entrypoint;
3. guided-staged through Playwright using
   `playwright.staging.config.ts`, zero retries, and the real `/guided/start`
   protocol.

The staged proof must mention the already-requested two LLMs during source
review using the fixed redundant sentence in Plan 07 and verify the assistant
says to wait, persists the intent across reload, and consumes it later without
new topology detail.

The executed pipeline must contain two independent provider-backed LLM nodes,
parallel per-row branches, a real require-all union coalesce, exact-field
cleanup, one ten-object JSON-array success artifact, and a distinct empty
JSON-array failure artifact. Prove twenty successful runtime provider
assessments, ten completed coalesces, exact row identity, value types/ranges,
semantic smoke checks, closed accounting, no fake/cache/replay substitution,
no handoff, no manual graph correction, and at most one automatic validation
repair.

Use `superpowers:systematic-debugging` for every failure. Fix the owning code or
skill surface, add a regression, redeploy a new integrated revision, and rerun
the affected proof. If an architecture/capability change occurs, rerun all three
proofs so the final evidence binds to one revision.

## Completion and handoff

Do not close `elspeth-7e2dd67275` until:

- all seven plan ledgers are green;
- the full deterministic 27-case matrix and generated-DAG parity gate pass;
- tutorial planner/schema/catalog/capability identity passes;
- the linear authoring implementation is absent;
- staging runs epoch 28/current schema after fresh state recreation;
- all three strict live proofs pass on one revision;
- every in-scope defect found during live work has a regression and is fixed;
- user and operator documentation describe capability parity and forward-only
  state recreation accurately.

Then close the controlling issue with the accepted commit, deployed revision,
test commands, live session/run identifiers, evidence directory, and a concise
statement that Wardline was intentionally not a mandatory gate for this work.
Use the normal workflow transition and expected assignee; do not force-close:

```bash
filigree close elspeth-7e2dd67275 \
  --expected-assignee "$AGENT_ID" \
  --actor "$AGENT_ID" \
  --commit "release/0.7.1@$(git rev-parse HEAD)" \
  --reason "All seven composer-parity slices, deployed three-surface live acceptance, and evidence gates passed; Wardline intentionally excluded as a mandatory gate by operator direction."
```

Your final report must distinguish:

1. code complete;
2. locally verified;
3. deployed;
4. live accepted;
5. tracker closed.

If any state is not achieved, state exactly what remains and continue working
while safe in-scope progress is possible.

---
