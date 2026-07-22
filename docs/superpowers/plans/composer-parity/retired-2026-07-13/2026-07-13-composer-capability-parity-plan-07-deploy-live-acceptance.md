# Composer Capability Parity Plan 07: Deploy and Live Acceptance

> **RETIRED (2026-07-17): DO NOT EXECUTE.** See
> [the current disposition](../2026-07-17-current-plan-disposition.md).

**Goal:** Recreate pre-release staging state, deploy the single canonical
composer implementation, and prove that freeform, guided-full, and
guided-staged independently derive and execute the ten-row two-LLM colour
split/merge pipeline.

**Depends on:** Plans 01-06 green on one integrated revision.

## Acceptance environment

Use environment variables; never commit credentials or include them in evidence:

```bash
export ELSPETH_TEST_BASE_URL='https://elspeth.foundryside.dev'
export ELSPETH_TEST_USERNAME='<staging username>'
export ELSPETH_TEST_PASSWORD='<staging password>'
export ELSPETH_PARITY_REVISION="$(git rev-parse HEAD)"
export ELSPETH_PARITY_EVIDENCE_DIR="$PWD/artifacts/composer-parity"
```

The operator-supplied values are injected only at execution time. Screenshots,
traces, logs, shell history excerpts, and JSON evidence must be checked for
credentials, cookies, authorization headers, provider secrets, and raw model
responses before retention.

The exact input fixture is
`evals/composer-parity/fixtures/two_llm_colour.csv`, SHA-256:

```text
067f0ffeb6a349fc33c1ce2f65cac65dcb37eb3bdd30ef8ca2670439238ba702
```

The exact initial request is
`evals/composer-parity/fixtures/two_llm_colour_request.txt`, SHA-256:

```text
37562b0fcfad56182dd33b3b72457681959ffa71f159beabcd387170187987d2
```

All three proofs must submit those exact request bytes as the first operator
intent. Record a hash of the complete operator transcript as well. Guided
review/approval responses may select or confirm proposed facts but may not add
topology or configuration detail.

## Required topology and result

```text
colour CSV source (10 rows)
          |
    explicit row fork
       /          \
 blue LLM       red LLM
       \          /
  require-all union coalesce
          |
 exact-field cleanup/mapper
          |
 success JSON array (10 rows)

source/LLM/coalesce/cleanup failures ---> failure JSON output (0 rows)
```

The request may say “two LLMs,” ask for JSON, ask to remove extra fields, and
name suitable plugins. It must not name composer tools, prescribe call order,
provide tool arguments, or import prepared YAML. The primary score permits one
automatic structured-validation repair and no operator correction after the
initial intent.

The success artifact contains exactly:

```text
color_name, hex,
blue_amount, blue_confidence, blue_reason,
red_amount, red_confidence, red_reason
```

Amounts are true integers in `0..100`; confidences are finite non-boolean
numbers in `0..1`; reasons are non-empty strings. Every one of the ten input
identities appears exactly once. Raw responses, usage, model metadata, branch
bookkeeping, and helper fields are absent.

Semantic smoke checks:

- Pure Blue and Navy: blue amount > red amount;
- Pure Red and Orange: red amount > blue amount.

## Task 1: Freeze the integrated candidate

- [ ] Record exact commit SHA, package version, frontend asset/build id, fixture
  hash, exact request hash, canonical schema hash, capability-core hash, and test
  timestamp.
- [ ] Run Plans 01, 05, and 06 gates plus the full guided/backend/frontend
  regression set on that exact SHA.
- [ ] Assert no active symbol, API arm, TypeScript arm, prompt, or setting exposes
  the removed linear-chain authoring implementation.
- [ ] Assert `SESSION_SCHEMA_EPOCH == 28` and
  `GUIDED_SESSION_SCHEMA_VERSION == 8`.
- [ ] Run `git diff --check` and confirm the intended tree state.

Any failure returns to the owning implementation plan. Do not deploy a partial
candidate.

## Task 2: Update operator and user documentation

Modify:

- `docs/runbooks/staging-session-db-recreation.md`
- `docs/guides/user-manual.md`
- composer eval/operator documentation.

- [ ] Explain that guided and freeform differ only in interaction and review;
  both support multiple sources/outputs, gates, queues, forks/coalesces,
  aggregations, row expansion, structured LLMs, and failure routes.
- [ ] Describe tutorial as a guided profile using the same planner and pipeline
  language.
- [ ] Document the wrong-stage behavior: wait and remember for a future stage,
  stable-id back/edit for a completed stage, disambiguation for ambiguous names,
  and a distinct unavailable-plugin error.
- [ ] Document the epoch-28 pre-release state recreation procedure and the fact
  that resumable sessions/tutorial progress are cleared.
- [ ] Replace any active old-source/database-restore procedure with the
  fix-current-code, recreate-fresh-state, redeploy, and reverify procedure. An
  archive may be retained briefly for diagnostics only and is never a supported
  service recovery point.
- [ ] Add a docs contract test that fails if the manual recommends switching to
  freeform for any supported topology.
- [ ] Run the staging-policy contract guard from Plan 03 so a source-ref switch,
  database restore command, or operational downgrade section cannot return to
  the active runbook.

Run:

```bash
uv run pytest tests/unit/docs/test_composer_capability_docs.py tests/unit/docs/test_staging_session_recreation_policy.py -q
```

## Task 3: Recreate staging state and deploy one implementation

- [ ] Put staging into its documented maintenance/drain state.
- [ ] Record only non-secret diagnostic metadata needed to identify the old
  state; do not preserve it as a supported input to the new build.
- [ ] Follow `docs/runbooks/staging-session-db-recreation.md` for the configured
  SQLite or PostgreSQL session store.
- [ ] Deploy the integrated candidate using the repository's authoritative
  staging deployment procedure and record the immutable revision/image digest.
- [ ] Verify health/readiness, epoch/current-schema startup, empty session state,
  login, catalog discovery, provider/model availability, and a fresh
  `/guided/start` session.
- [ ] Confirm the deployed frontend asset/build id matches the candidate.

Execute the guarded forward-only staging procedure created in Plan 03; every
command exits nonzero on failure and the recreation script has no source/data
restore arm:

```bash
test -z "$(git status --porcelain)"
test "$(git rev-parse HEAD)" = "$ELSPETH_PARITY_REVISION"
test -x scripts/recreate-staging-session-state.sh
uv sync --frozen --all-extras
(cd src/elspeth/web/frontend && npm ci && npm run build)
sudo scripts/recreate-staging-session-state.sh \
  --expected-revision "$ELSPETH_PARITY_REVISION" \
  --confirm RECREATE
curl --unix-socket /run/elspeth/uvicorn.sock -fsS http://localhost/api/health
curl -fsS "$ELSPETH_TEST_BASE_URL/api/health"
```

The script is the executable authority for safe DB-path resolution, service
drain/stop, pre-release state deletion, restart, epoch-28 verification, and
fresh-session smoke. Save its sanitized output under the revision evidence
directory.

If startup or smoke verification fails, use systematic debugging: capture the
exact symptom, reproduce with the narrowest safe check, trace to the responsible
boundary, add a failing regression, fix the canonical implementation, deploy a
new integrated revision, and restart this task from state recreation.

## Task 4: Build and regression-test the strict live evidence recorder

Create:

- `evals/composer-parity/live_acceptance.py` — nonzero-on-failure live runner and
  `verify-existing` CLI;
- `tests/unit/evals/composer_parity/test_live_acceptance.py` — parameterized
  oracle and negative controls;
- `tests/integration/evals/composer_parity/test_live_acceptance_server.py` —
  authenticated freeform/guided-full evidence capture;
- `src/elspeth/web/frontend/tests/e2e/composer-capability-parity.spec.ts` — staged
  browser journey and raw evidence export.

All three surfaces write under
`$ELSPETH_PARITY_EVIDENCE_DIR/$ELSPETH_PARITY_REVISION/`. The verifier rejects
missing surfaces, mixed revisions/builds, a revision that differs from the
deployed health/build response, or evidence outside that run directory.

Create a recorder that binds each proof to:

- deployed revision/image and frontend build id;
- surface/profile and session/run ids;
- input fixture hash and exact initial request hash;
- actual per-call rendered prompt, capability-core, canonical schema, effective
  tool-schema, catalog, and plugin-assistance hashes;
- proposal draft/base/anchor hashes and repair count;
- committed composition version, normalized graph digest, public YAML hash, and
  validation result;
- terminal run accounting, sanitized audit references, success/failure artifact
  hashes, and elapsed times;
- Playwright screenshots/trace for guided-staged.

The recorder must fail unless all of these are true:

- two distinct LLM nodes exist, not one node with two queries;
- both branches receive all ten source rows;
- the merge is a real `require_all` union coalesce;
- relevant error routes reach the failure output;
- cleanup emits exactly the eight fields;
- success artifact has ten rows and failure artifact has zero;
- type/range/identity/semantic checks above pass;
- terminal accounting reports ten succeeded rows, zero failed rows, zero pending
  tokens, and closed integrity;
- audit/lineage shows ten logical blue assessments, ten logical red assessments,
  and ten completed coalesces; provider retries are reported separately;
- each of the twenty runtime branch assessments has a successful audited
  provider completion with non-empty provider and model identifiers; planner
  calls are classified separately, providers/models marked fake, mock, test, or
  replay are rejected, and cache/replay hits do not satisfy the primary proof;
- no handoff, YAML import, recipe substitute, graph correction, operator
  correction, timeout, or second automatic repair occurred.

Parse the success artifact and require one JSON array root of length ten, not
JSONL or separate objects. Require exact field-set equality on every object.
Require a distinct failure artifact whose JSON root is an empty array.

The recorder must reject self-reported planner metadata when it differs from the
wrapped completion request or server audit evidence.

Add one parameterized negative fixture for every rejection condition above,
including wrong request hash, transcript with added choreography, one LLM node,
missing branch row, wrong merge mode, unrouted error, extra field, JSONL root,
missing/nonnull failure artifact, duplicate identity, bool-as-number, NaN,
semantic-smoke failure, non-terminal accounting, missing provider evidence,
fake provider, cache/replay, mixed revision, timeout, correction, and excess
repair. Each case must make the CLI exit nonzero and name the failed invariant.

Run:

```bash
uv run pytest tests/unit/evals/composer_parity/test_live_acceptance.py tests/integration/evals/composer_parity/test_live_acceptance_server.py -q
```

## Task 5: Run freeform big-bang proof

- [ ] Start from a clean authenticated session and submit the committed plain-
  English request bytes through the normal production compose route/browser
  surface.
- [ ] Do not mention composer tools or setting-level choreography.
- [ ] Accept only a valid runnable proposal produced within 180 seconds and at
  most one automatic repair.
- [ ] Execute the pipeline; require terminal state within 300 seconds.
- [ ] Download/read both artifacts and run the strict evidence recorder.
- [ ] Save sanitized evidence as `freeform_big_bang`.

Exact command; it exits nonzero on any oracle failure:

```bash
uv run python evals/composer-parity/live_acceptance.py run \
  --surface freeform_big_bang \
  --base-url "$ELSPETH_TEST_BASE_URL" \
  --fixture evals/composer-parity/fixtures/two_llm_colour.csv \
  --intent evals/composer-parity/fixtures/two_llm_colour_request.txt \
  --revision "$ELSPETH_PARITY_REVISION" \
  --evidence-dir "$ELSPETH_PARITY_EVIDENCE_DIR"
```

## Task 6: Run guided-full proof

- [ ] Call the deployed authenticated guided-full server entrypoint with the same
  initial request and fixture; do not call `plan_pipeline()` directly.
- [ ] Require the server path to perform request parsing, prompt/tool/catalog
  assembly, proposal validation, audited commit, and execution.
- [ ] Apply the same 180-second planning, one-repair, 300-second execution, and
  strict artifact/topology/audit oracle.
- [ ] Save sanitized evidence as `guided_full`.

Exact command; it exits nonzero on any oracle failure:

```bash
uv run python evals/composer-parity/live_acceptance.py run \
  --surface guided_full \
  --base-url "$ELSPETH_TEST_BASE_URL" \
  --fixture evals/composer-parity/fixtures/two_llm_colour.csv \
  --intent evals/composer-parity/fixtures/two_llm_colour_request.txt \
  --revision "$ELSPETH_PARITY_REVISION" \
  --evidence-dir "$ELSPETH_PARITY_EVIDENCE_DIR"
```

## Task 7: Run guided-staged Playwright proof

- [ ] Use the repository Playwright skill and tests under
  `src/elspeth/web/frontend/tests/e2e` against `ELSPETH_TEST_BASE_URL`.
- [ ] Log in from environment credentials; never embed them in the test.
- [ ] Start at `/guided/start` and provide the exact committed intent. During
  source review send exactly this fixed, semantically redundant probe:
  `Please keep the two independent LLM assessments from my original request.`
  It adds no plugin, topology, call-order, or setting information.
- [ ] Assert the assistant says to wait for the transformation stage, retains
  the intent across a page reload, does not advance or configure it early, and
  consumes it when transformation/topology review begins.
- [ ] Review the complete DAG, accept it, execute it, and inspect/download both
  artifacts.
- [ ] Require wire-ready within 360 seconds, execution within 300 seconds, and
  Playwright retries set to zero.
- [ ] Run the strict recorder and save sanitized screenshots/trace/evidence as
  `guided_staged`.

Exact commands; Playwright retries are zero and `verify-existing` exits nonzero
on any oracle failure:

```bash
cd src/elspeth/web/frontend
STAGING_BASE_URL="$ELSPETH_TEST_BASE_URL" \
STAGING_USERNAME="$ELSPETH_TEST_USERNAME" \
STAGING_PASSWORD="$ELSPETH_TEST_PASSWORD" \
PLAYWRIGHT_BACKEND_BASE_URL="$ELSPETH_TEST_BASE_URL" \
npx playwright test \
  --config=playwright.staging.config.ts \
  tests/e2e/composer-capability-parity.spec.ts \
  --project=chromium \
  --retries=0
cd ../../../..
uv run python evals/composer-parity/live_acceptance.py verify-existing \
  --surface guided_staged \
  --revision "$ELSPETH_PARITY_REVISION" \
  --evidence-dir "$ELSPETH_PARITY_EVIDENCE_DIR"
```

## Task 8: Fix-forward loop for any live defect

For every error, timeout, invalid proposal, wrong-stage mistake, audit mismatch,
or output mismatch:

1. preserve the exact sanitized symptom and first failing boundary;
2. reproduce through the narrowest deterministic or staging test;
3. trace the data path before editing;
4. write a failing regression in the owning plan's suite;
5. fix the shared canonical implementation or guidance at the source;
6. rerun the narrow test, owning plan gate, Plan 06 parity gate, and affected
   live proof on the newly deployed revision;
7. after any architecture/capability change, rerun all three live proofs so the
   final evidence set binds to one revision.

Do not restore the removed authoring path or add a switch around the defect.

## Final acceptance

- [ ] One exact deployed revision owns all evidence.
- [ ] Freeform big-bang passes the strict live oracle.
- [ ] Guided-full passes the strict live oracle through the deployed server.
- [ ] Guided-staged passes the strict live oracle through Playwright, including
  wait/persist/consume behavior for early LLM intent.
- [ ] All three normalized graphs and business-output contracts are equivalent.
- [ ] `verify-existing --surface all` passes for one evidence directory and
  rejects mixed revision/provider/request evidence.
- [ ] Documentation and state-recreation runbook are current.
- [ ] No secrets/raw model responses are present in retained evidence.
- [ ] All in-scope live defects have regressions and are fixed on the accepted
  revision.
- [ ] The controlling Filigree issue is closed only now, provided every other
  acceptance criterion passes; the temporarily removed Wardline gate is not
  treated as mandatory.
