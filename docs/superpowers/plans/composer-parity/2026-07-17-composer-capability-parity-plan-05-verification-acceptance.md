# Composer Capability Parity Plan 05: Verification and Staging Acceptance

**Goal:** Make capability regressions fail deterministically and prove the
integrated implementation on staging with the two-LLM colour split/merge.

**Architecture:** A nine-class canonical corpus drives three independent
production adapters through real prompt/tool parsing, candidate validation,
proposal persistence, audited commit, and runtime compilation. Hypothesis adds
bounded generated valid DAGs. Staging reuses the repository deployment and
session-recreation runbooks and retains only evidence needed to verify the
accepted revision and outputs.

**Prerequisite:** Plans 01-04 pass on one integrated revision. Task 1 below is
required before the parity corpus or staging acceptance is evaluated.

## Task 1: Type public structured LLM query discovery

**Files:**

- Modify: `src/elspeth/plugins/transforms/llm/base.py`
- Modify: `src/elspeth/plugins/transforms/llm/multi_query.py`
- Modify: `src/elspeth/plugins/transforms/llm/transform.py`
- Modify: `tests/unit/plugins/llm/test_llm_config.py`
- Modify: `tests/unit/web/catalog/test_service.py`
- Modify: `tests/unit/contracts/test_plugin_assistance_coverage.py`
- Modify: `tests/integration/web/test_catalog_discovery.py`

- [ ] Replace `LLMConfig.queries` untyped nested dictionaries with Pydantic
  query/output-field configuration models. Preserve both accepted authoring
  forms—mapping keyed by query name and list entries carrying `name`—and
  normalize them to the same runtime query specs without changing execution.
- [ ] Adapt `resolve_queries()` and every LLM config/template/required-field
  validator to consume the typed models. Preserve duplicate-name, reserved
  suffix, output collision, enum/type, template, and pass-through guarantees;
  malformed input must still fail closed with safe configuration errors.
- [ ] Assert the generated catalog schema and lowered discovery response expose
  query `name`, `input_fields`, `template`, `response_format`, `output_fields`,
  field `suffix`/`type`, and applicable pass-through/collision behavior. A
  generic `additionalProperties: true` object does not satisfy this gate.
- [ ] Add one concise structured multi-query example to the shared LLM plugin
  assistance. Catalog, freeform, guided-full, guided-staged, and tutorial all
  consume that same assistance; do not copy the example into a guided prompt.
- [ ] Keep the already-landed section 5.3 probe prerequisite intact: commit
  `a718a39ff` prepares detached validation options in the exact order
  `deep_thaw` → `strip_authoring_options` →
  `redact_secret_refs_for_validation`. Expected configuration-probe failures
  remain in the existing redacted safe failure category and unexpected
  framework failures still propagate.

Run:

```bash
uv run pytest \
  tests/unit/plugins/llm/test_llm_config.py \
  tests/unit/web/catalog/test_service.py \
  tests/unit/contracts/test_plugin_assistance_coverage.py \
  tests/integration/web/test_catalog_discovery.py -q
```

Expected: both query input forms validate and normalize identically, and public
catalog discovery advertises the complete typed structured-output contract.

## Task 2: Create the canonical fixture corpus

**Files:**

- Create: `evals/composer-parity/README.md`
- Create: `evals/composer-parity/fixtures/linear_transform.json`
- Create: `evals/composer-parity/fixtures/conditional_gate.json`
- Create: `evals/composer-parity/fixtures/multi_output.json`
- Create: `evals/composer-parity/fixtures/fork_coalesce.json`
- Create: `evals/composer-parity/fixtures/multi_source_queue.json`
- Create: `evals/composer-parity/fixtures/aggregation.json`
- Create: `evals/composer-parity/fixtures/row_expansion.json`
- Create: `evals/composer-parity/fixtures/error_routing.json`
- Create: `evals/composer-parity/fixtures/structured_llm.json`
- Create: `evals/composer-parity/fixtures/two_llm_colour.csv`
- Create: `evals/composer-parity/fixtures/two_llm_colour_request.txt`
- Create: `tests/unit/evals/composer_parity/test_fixtures.py`
- Modify: `.gitignore`

- [ ] Each JSON fixture stores a plain-language intent, canonical accepted
  arguments, semantic node/edge/output expectations, and runtime assertions.
  It must not script composer tool-call order.
- [ ] Cover linear transform, conditional gate, multiple outputs,
  fork/coalesce, multi-source queue, aggregation, row expansion, error routing,
  and structured LLM output consumed downstream.
- [ ] Materialize the exact ten-row colour CSV and outcome-only request from the
  retained design. Record their SHA-256 values in `README.md` after creation;
  the hash protects test input integrity, not a plan version.
- [ ] Add a fixture loader that validates each canonical payload with
  `SetPipelineArgumentsModel` and the current policy catalog.
- [ ] Add narrow `.gitignore` negations for `/evals/composer-parity/` and its
  committed contents. Verify `git check-ignore` reports the corpus as included
  and `git ls-files` contains every Task-2 fixture; generated run evidence
  remains ignored under `output/playwright/composer-parity/`.

Run:

```bash
sha256sum evals/composer-parity/fixtures/two_llm_colour.csv
sha256sum evals/composer-parity/fixtures/two_llm_colour_request.txt
uv run pytest tests/unit/evals/composer_parity/test_fixtures.py -q
```

Expected: hashes match the values committed in the eval README and all nine
fixtures validate.

## Task 3: Build the three-surface real-path matrix

**Files:**

- Create: `tests/helpers/composer_graphs.py`
- Create: `tests/integration/web/composer/parity/conftest.py`
- Create: `tests/integration/web/composer/parity/test_fixture_matrix.py`
- Create: `tests/integration/web/composer/parity/test_repair_and_deferral.py`

- [ ] Implement three distinct request adapters:
  freeform calls the normal compose route/service; guided-full calls the
  authenticated `/guided/plan` endpoint; guided-staged starts and drives the
  persisted stage protocol through proposal acceptance. Each starts with the
  same intent and independently invokes the deterministic fake completion.
- [ ] Traverse real prompt/tool assembly, terminal parser, custody, candidate,
  durable proposal, acceptance, audited `set_pipeline`, immutable state, and
  public YAML compiler. Do not inject a `PipelineProposal` or
  `CompositionState`.
- [ ] Normalize generated ids, connection names, temporary output paths,
  versions, timestamps, rationale, and profile metadata. Preserve plugin/node
  kinds, normalized options, edge roles/routes, topology, policies, merge mode,
  field contracts, output schemas, and failure policy.
- [ ] Parameterize nine fixtures by three surfaces and assert graph isomorphism,
  equivalent validation, runtime graph, and public YAML semantics: 27 cases,
  no provider network, skips, or expected failures.
- [ ] Add one-repair success, repair exhaustion, policy rejection, future-stage
  retention across restart, completed-stage back/edit, unavailable plugin, and
  tutorial identity cases.

Run:

```bash
uv run pytest tests/integration/web/composer/parity -q
```

Expected: at least 27 positive surface/fixture cases pass independently, plus
the repair/deferral negatives.

## Task 4: Add generated-DAG and mutation controls

**Files:**

- Create: `tests/property/web/composer/test_authoring_surface_parity.py`
- Create: `tests/integration/web/composer/parity/test_schema_mutation_controls.py`
- Modify: `pyproject.toml` only if the required marker/dependency is absent.

- [ ] Generate bounded policy-valid canonical DAGs with plural sources/outputs,
  transforms, gates, aggregation, queues, fork/coalesce, and error routes using
  deterministic Hypothesis settings. Reject invalid combinations in the
  strategy; every emitted case must pass the canonical boundary.
- [ ] Pass each generated case independently through all three real-path
  adapters and compare committed runtime graphs by isomorphism. Shrinking must
  report intent, payload, surface, and first differing semantic attribute.
- [ ] Mutate one surface's actual advertised terminal schema immediately before
  completion: remove `fork_to`, remove `merge`, exclude `queue`, and cap named
  sources at one. Each control must fail the same parity assertion and name the
  field/surface; a dictionary-only comparison does not count.

Run:

```bash
uv run pytest \
  tests/property/web/composer/test_authoring_surface_parity.py \
  tests/integration/web/composer/parity/test_schema_mutation_controls.py -q
```

Expected: 50 deterministic generated examples pass and every deliberate
surface narrowing is detected.

## Task 5: Add the live acceptance oracle and frontend journey

**Files:**

- Create: `evals/composer-parity/live_acceptance.py`
- Create: `tests/unit/evals/composer_parity/test_live_acceptance.py`
- Create: `tests/integration/evals/composer_parity/test_live_acceptance_server.py`
- Create: `src/elspeth/web/frontend/tests/e2e/composer-capability-parity.staging.spec.ts`
- Modify: `src/elspeth/web/frontend/src/test/a11y/components.a11y.test.tsx`
- Modify: `docs/guides/user-manual.md`
- Modify: `docs/runbooks/staging-session-db-recreation.md`
- Create: `tests/unit/docs/test_composer_capability_docs.py`

- [ ] The oracle accepts sanitized evidence for one deployed revision and
  verifies: two distinct LLM nodes; both receive all ten rows; real
  require-all union coalesce; error routes reach the failure output; exact
  eight-field cleanup; ten successes and zero failures; integer/range/reason
  contracts; unique input identities; and absence of raw responses, usage,
  model metadata, and branch bookkeeping.
- [ ] Require audit evidence for the exact proposal/commit/run and successful
  real provider completions. Keep planner calls separate; fake, mock, test,
  replay, and cache evidence cannot satisfy the live provider proof. Never
  retain credentials, cookies, authorization headers, resolved secrets, or raw
  provider responses.
- [ ] Add negative fixtures for one LLM, missing branch row, wrong merge,
  unrouted failure, extra/missing field, JSONL root, duplicate identity,
  bool-as-number, NaN, nonterminal accounting, fake provider, excess repair,
  operator topology correction, and mixed revision.
- [ ] Verify `git ls-files --error-unmatch
  evals/composer-parity/live_acceptance.py` succeeds after creating the oracle;
  do not rely on a broad ignore exception that can silently omit it.
- [ ] The Playwright journey drives `/guided/start`, sends an early reminder to
  retain the two independent LLM assessments, reloads, verifies wait/persist/
  consume behavior, reviews the complete graph, accepts, runs, and passes the
  same oracle.
- [ ] Add `ProposePipelineTurn` to the central axe suite for default, revise,
  stale, error, and tutorial states. Assert accessible names, live-region
  behavior, and focus placement after validation/stale transitions.
- [ ] Update user docs to state that guided/freeform differ in interaction, list
  supported structures, explain wrong-stage retention/back-edit, and describe
  tutorial as a shared-planner profile. Add a docs test rejecting advice to
  switch to freeform for a supported topology.

Run:

```bash
uv run pytest \
  tests/unit/evals/composer_parity/test_live_acceptance.py \
  tests/integration/evals/composer_parity/test_live_acceptance_server.py \
  tests/unit/docs/test_composer_capability_docs.py -q
cd src/elspeth/web/frontend
npm test -- --run src/components/chat/guided
npm run typecheck
```

Expected: PASS, including every negative control.

## Task 6: Recreate staging, deploy, and run all three proofs

**Authoritative docs:**

- `docs/runbooks/staging-session-db-recreation.md`
- `docs/runbooks/aws-ecs-deployment.md` or the deployment runbook for the
  actual staging shape

- [ ] On the exact integrated revision, run the full backend, testcontainer,
  property, frontend, and Playwright gates. Assert session epoch 30, guided
  schema 8, landscape epoch 28, and no active chain proposal path.
- [ ] Drain staging and recreate only the pre-release session store using the
  updated runbook. Verify health, schema probe, empty sessions, authentication,
  request-scoped catalog, provider/model availability, frontend build id, and a
  fresh guided start before admitting the acceptance runs.
- [ ] Run freeform, guided-full, and guided-staged from clean sessions with the
  same committed fixture and initial request. Allow at most one automatic
  structured repair and no operator topology/configuration correction.
- [ ] Pass each run through `live_acceptance.py`, then compare normalized graph
  and business-output contracts across all three. Keep sanitized evidence under
  one revision directory only as long as it supports diagnosis and acceptance.
- [ ] If a run fails, add a deterministic regression at the first failing
  shared boundary, fix the current implementation, rerun its phase gate, deploy
  the corrected revision, recreate fresh session state, and rerun all three
  proofs. Do not restore the deleted chain path or add an architecture switch.
- [ ] Create the ignored evidence directory with restrictive permissions,
  reject symlinks and files outside its resolved revision directory, and remove
  credentials/cookies/raw provider responses before retention or cleanup.

Example invocation shape; credentials remain environment-only:

```bash
uv run python evals/composer-parity/live_acceptance.py run \
  --surface freeform \
  --base-url "$ELSPETH_EVAL_BASE_URL" \
  --fixture evals/composer-parity/fixtures/two_llm_colour.csv \
  --intent evals/composer-parity/fixtures/two_llm_colour_request.txt \
  --revision "$(git rev-parse HEAD)" \
  --evidence-dir output/playwright/composer-parity
```

Repeat with `--surface guided_full`; run the staged Playwright test with
environment credentials and then verify its exported evidence with the same
oracle.

Final gate:

```bash
uv run ruff check src tests evals
uv run ruff format --check src tests evals
uv run mypy src
uv run python scripts/check_contracts.py
uv run python scripts/cicd/generate_skill_inventory.py --check
uv run pytest \
  tests/unit/elspeth_lints/test_composer_rules.py \
  tests/unit/web/composer/test_adequacy_guard.py \
  tests/unit/web/composer/test_redaction_completeness_property.py \
  tests/unit/web/composer/test_tool_redaction_policy.py -q
uv run pytest -q
uv run pytest tests/testcontainer/web tests/property -q
cd src/elspeth/web/frontend
npm run typecheck
npm test -- --run
npm run build
npx playwright test \
  --config=playwright.staging.config.ts \
  tests/e2e/composer-capability-parity.staging.spec.ts \
  --retries=0
cd ../../../..
git diff --check
```

Expected: every command exits 0 and all three staging proofs pass on one
revision.

**Definition of done:** The deterministic and generated suites detect any
surface narrowing, all three deployed ordinary authoring surfaces independently
derive and execute the colour graph, tutorial retains shared mechanical
identity, retained evidence is redacted and internally consistent, and the
controlling issue can close.
