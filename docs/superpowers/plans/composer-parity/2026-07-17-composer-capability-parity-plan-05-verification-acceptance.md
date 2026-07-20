# Composer Capability Parity Plan 05: Verification and Staging Acceptance

**Goal:** Make capability regressions fail deterministically and prove the
integrated implementation on staging with the two-LLM colour split/merge.

**Architecture:** A nine-class canonical corpus drives three independent
production adapters through real prompt/tool parsing, candidate validation,
proposal persistence, audited commit, and runtime compilation. Hypothesis adds
bounded generated valid DAGs. Staging reuses the repository deployment and
session-recreation runbooks and retains only evidence needed to verify the
accepted revision and outputs.

**Prerequisite:** Plans 01-04 pass on one integrated revision. Task 0
(prerequisite fix) and Task 1 both land before Task 3's negative cases and
before the parity corpus / staging acceptance are evaluated.

**Execution ordering (single source of truth):**

- **Task 0** (freeform planner-failure HTTP translation, below) precedes
  **Task 3** — it gates only Task 3's repair-exhaustion / policy-rejection
  negative cases and Task 5's closed-audit-trail oracle, not the 27 positive
  isomorphism cases.
- **Task 1** (typed LLM query discovery) precedes **Task 3** — the
  `structured_llm` and two-LLM colour fixtures cannot pass candidate validation
  until the typed queries and the §5.3 secret-reference probe fix are in place.
- **Task 2 → Task 3 → Task 4** is a hard chain. Task 4's generated-DAG
  comparison reuses Task 3's real-path adapters and the
  `tests/helpers/composer_graphs.py` isomorphism helper, and its strategy's
  canonical-boundary check reuses Task 2's `SetPipelineArgumentsModel` + policy
  loader. None of these exist at HEAD `72a7efc0f`; Task 4 cannot be built
  standalone.
- **Task 5** depends on Task 2 artifacts (the `evals/composer-parity/`
  directory, its `.gitignore` negations, and the
  `tests/unit/evals/composer_parity/` package) plus a new
  `tests/integration/evals/` package; do not start Task 5 standalone.

## Cross-cutting constraints

These apply across every task below.

- **`check_contracts` is a known-RED baseline (elspeth-322b6c6f2a).** The
  Plan 05 obligation is "introduce **no new** findings under `src/elspeth`",
  **not** `check_contracts` exit 0. The ~26-34 accumulated `dict[str, Any]` /
  type-definition violations are pre-existing, concentrated in web/composer and
  web/execution, and the operator ordered NO whitelisting. Task 6's final gate
  is adjusted accordingly: treat `scripts/check_contracts.py` as a
  changed-files-delta / no-new-findings check, not an exit-0 command.
- **Signed artifacts stay untouched during Plan 05 and reconcile only at the
  release-merge boundary** via elspeth-judge staging + operator sign: the
  tier-model allowlists (`config/plugins.yaml`), the fingerprint baseline, and
  the `transform.py` PH3 hash. Do not re-sign, rotate, or hand-edit any of them
  in this plan; ignore signature CI churn mid-release. **Exception:** Task 1's
  de-dict makes several `llm` `allowed_dict_patterns` entries in
  `config/cicd/contracts-whitelist.yaml` stale (that whitelist is **not** a
  signed artifact); prune those in lockstep with the signature changes, and
  introduce **no new** `dict[str, Any]` param/return in the typed-model helpers.
- **Two false-green traps must be defeated where they occur:**
  1. *Task 4 mutation controls* must trip the **schema-identity gate**
     `build_planner_capability_manifest` (`capability_skill.py:188`; hash-compares
     `advertised_schema` vs `canonical_schema` at `:223` and raises
     `AuditIntegrityError`), **not** the graph-isomorphism assertion. Under a
     scripted fake completion a narrowed advertised schema still produces an
     identical committed graph, so isomorphism cannot detect the narrowing —
     routing the controls through it yields controls that fail to fail.
  2. *Task 3's freeform adapter* must bypass or guarantee non-match of the
     server recipe fast-path (`service.py:2534` `prepare_pipeline_plan`, which
     runs before `plan_pipeline` at `service.py:2567`) so freeform provably
     traverses `plan_pipeline` and `build_planner_capability_manifest` — else
     parity compares a recipe-router graph against LLM-planner graphs.

## Task 0 (prerequisite fix): Freeform planner-failure HTTP translation — elspeth-54c11243a3 (still-live half)

**Files:**

- Modify: `src/elspeth/web/sessions/routes/composer/compose.py`
- Modify: `src/elspeth/web/sessions/routes/messages.py`
- Modify: `src/elspeth/web/composer/service.py` (only if a freeform-applicable
  failure-disposition writer is needed alongside the route translation)
- Create/Modify: a freeform planner-failure regression under
  `tests/integration/web/composer/`

- [ ] **Defect:** `PipelinePlannerError` (bare `RuntimeError` subclass,
  `src/elspeth/web/composer/pipeline_planner.py:84`) is translated by **nobody**
  on the freeform empty-pipeline path. `compose()`
  (`src/elspeth/web/composer/service.py:1938`) handles only
  `ComposerConvergenceError` (L2018), `ComposerPluginCrashError` (L2025), and
  `(ComposerServiceError, LiteLLMAPIError)` (L2078); `send_message`
  (`routes/messages.py:93`) and `recompose` (`routes/composer/compose.py:73`)
  catch the same families but not `PipelinePlannerError`; there is no app-level
  handler. Only the guided path translates it
  (`routes/composer/guided_plan.py:89`,
  `if isinstance(exc, PipelinePlannerError)`). A freeform planner failure
  (COST_CAP_EXCEEDED / MALFORMED_RESPONSE / TIMEOUT) therefore escapes as a
  generic 500 with no "failed" progress event and no closed
  failure-disposition record.
- [ ] **Fix:** add `PipelinePlannerError` → safe-response + "failed"
  `ComposerProgressEvent` translation to the freeform routes (`compose.py`
  `recompose` and `messages.py` `send_message`), **and** a durable
  terminal/closed failure-disposition record mirroring the guided path's
  `fail_guided_operation_with_audit` (defined
  `src/elspeth/web/sessions/service.py:4367`; protocol
  `src/elspeth/web/sessions/protocol.py:2173`; invoked for guided-full at
  `routes/composer/guided_plan.py:318` and `:346`).
- [ ] **DO NOT RE-TOUCH — the audit-evidence half is ALREADY FIXED.**
  `plan_pipeline` stamps `recorder.llm_calls` onto the escaping exception via
  `attach_llm_calls(exc, recorder, start_index=llm_call_start)`
  (`pipeline_planner.py:881`), and `_plan_and_stage_empty_pipeline`
  (`service.py:2456`) persists it before re-raise. The Task 3 route has **no**
  recorder handle and must **not** re-persist this.
- [ ] **Scope:** Task 0 gates ONLY Task 3's repair-exhaustion / policy-rejection
  negative cases and Task 5's closed-audit-trail oracle — NOT the 27 positive
  isomorphism cases.

Run:

```bash
uv run pytest tests/integration/web/composer -q -k "planner_failure or freeform"
```

Expected: a freeform planner failure returns a safe response with a "failed"
progress event and lands a closed failure-disposition record, matching the
guided path.

## Task 1: Type public structured LLM query discovery

**Files:**

- Modify: `src/elspeth/plugins/transforms/llm/base.py`
- Modify: `src/elspeth/plugins/transforms/llm/multi_query.py`
- Modify: `src/elspeth/plugins/transforms/llm/transform.py`
- Modify: `src/elspeth/web/catalog/knob_schema.py` (Ruling B: extend
  `_lower_field` / `_unwrap_optional` so lowered discovery exposes the typed
  query structure — the larger sub-task)
- Modify: `config/cicd/contracts-whitelist.yaml` (prune now-stale `llm`
  `allowed_dict_patterns` entries in lockstep; introduce no new `dict[str, Any]`)
- Modify: `tests/unit/plugins/llm/test_llm_config.py`
- Modify: `tests/unit/plugins/llm/test_multi_query.py`
- Modify: `tests/unit/web/catalog/test_service.py`
- Modify: `tests/unit/contracts/test_plugin_assistance_coverage.py`
- Modify: `tests/integration/web/test_catalog_discovery.py`

- [ ] Replace `LLMConfig.queries` untyped nested dictionaries
  (`list[dict[str, Any]] | dict[str, dict[str, Any]] | None`, `base.py:68`) with
  a typed model. **Reuse** the existing `OutputFieldConfig`, `ResponseFormat`,
  and `OutputFieldType` (already Pydantic / StrEnum in `multi_query.py`); only a
  **query-level** Pydantic model is net-new. Keep `QuerySpec` as the frozen
  runtime spec. Preserve both accepted authoring forms—mapping keyed by query
  name (value carries no `name`) and list entries carrying `name`—and normalize
  them to the same runtime query specs without changing execution. Note the
  dual-form asymmetry under `extra=forbid`: a mapping value must not require
  `name` while a list entry must; add a dual-form round-trip test asserting both
  produce identical `QuerySpec` lists.
- [ ] Adapt `resolve_queries()` and every LLM config/template/required-field
  validator to consume the typed models. Preserve duplicate-name, reserved
  suffix, output collision, enum/type, template, and pass-through guarantees;
  malformed input must still fail closed with safe configuration errors.
- [ ] **Discovery gate — both surfaces required (Ruling B).**
  (1) `json_schema` via `LLMTransform.get_config_schema` emits `$defs` for the
  query/output-field models automatically once `queries` is typed (mechanical):
  assert `PluginSchemaInfo.json_schema['$defs']` exposes query `name`,
  `input_fields`, `template`, `response_format`, `output_fields`, and field
  `suffix`/`type`. (2) **Also** extend the lowered `knob_schema`
  (`_lower_field` / `_unwrap_optional`, `web/catalog/knob_schema.py:170` / `:117`)
  so the lowered discovery response exposes the same typed query structure
  rather than falling through to a generic `json-value` — this needs union +
  nested-model recursion in `_lower_field` and is the **larger sub-task**. A
  generic `additionalProperties: true` object does not satisfy this gate.
  Cross-query output-collision and pass-through / guaranteed-field behavior are
  **runtime rules** (`resolve_queries` / `_build_multi_query_output_schema`),
  not properties of the input `queries` schema — assert them in the
  validation / output-schema tests, not in this schema-shape gate.
- [ ] Add one concise structured multi-query example to the shared LLM plugin
  assistance. Catalog, freeform, guided-full, guided-staged, and tutorial all
  consume that same assistance; do not copy the example into a guided prompt.
- [ ] **Safe-failure placement (Ruling A).** Today the cross-query validation
  (duplicate-name, reserved-suffix, output-collision, positional-var) raises a
  **bare `ValueError`** inside `resolve_queries` at `__init__` time
  (`multi_query.py:142`; e.g. the "Duplicate query name" raise at `:223`), and
  `OutputFieldConfig` raises a pydantic `ValidationError` — neither is wrapped
  by `from_dict` nor matched by `_is_config_probe_exception`, so both
  **propagate** today rather than landing in the §5.3 redacted-safe category.
  **Relocate** the cross-query validation into an `LLMConfig` pydantic
  `model_validator` that runs at config parse / validation time, so failures
  surface as `PluginConfigError` (the redacted-safe category per §5.3).
  `QuerySpec` runtime resolution stays; only VALIDATION moves earlier.
  **Guard:** the relocated validation must be specific to config-shape errors
  and must NOT swallow genuine framework failures. Add a probe-classification
  regression covering malformed structured queries.
- [ ] Keep the already-landed section 5.3 probe prerequisite intact: commit
  `a718a39ff` prepares detached validation options in the exact order
  `deep_thaw` → `strip_authoring_options` →
  `redact_secret_refs_for_validation`. Expected configuration-probe failures
  remain in the existing redacted safe failure category and unexpected
  framework failures still propagate. (Preserving the redacted-safe category
  for structured-query errors is **not** automatic — it requires the Ruling A
  relocation above.)

Run:

```bash
uv run pytest \
  tests/unit/plugins/llm/test_llm_config.py \
  tests/unit/plugins/llm/test_multi_query.py \
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
- [ ] Materialize the ten-row colour CSV and outcome-only request. The
  **verbatim source** for both is the run-sheet
  (`docs/superpowers/plans/2026-07-13-two-llm-colour-hybrid-demo-run-sheet.md`,
  CSV at L70-82, request at L141-148), **not** the two-LLM colour hybrid design
  doc (`docs/superpowers/specs/2026-07-13-two-llm-colour-hybrid-pipeline-design.md`,
  which holds the CSV only as a markdown table and merely describes the request);
  keep that design doc as the authority for the 10 colour/hex values and the
  8-field hybrid contract. Use the
  consistent fixture filenames `two_llm_colour.csv` and
  `two_llm_colour_request.txt`. **Filename normalization (Ruling C):** the
  run-sheet request opens "I've uploaded two-llm-colour-palette.csv"; normalize
  that in-prose filename token to `two_llm_colour.csv` for corpus consistency
  (accepting the request is no longer byte-verbatim). **Byte canonicalization
  (Ruling C):** `README.md` must pin **LF newlines, UTF-8 with no BOM, and
  exactly one trailing newline** for both files, and the recorded SHA-256 must
  be computed against that pinned form (otherwise the hash gate is a
  platform-dependent flake). Record both SHA-256 values in `README.md` after
  creation; the hash protects test input integrity, not a plan version.
- [ ] Add a fixture loader that validates each canonical payload structurally
  with `SetPipelineArgumentsModel.model_validate`
  (`src/elspeth/web/composer/redaction.py`) **plus** plugin-availability against
  the trained-operator snapshot (`PolicyCatalogView.for_trained_operator` and
  its availability lookups). **Defer** full committed-graph validation
  (`validate_composition_state`) to Task 3's real-path matrix — the
  args→`CompositionState` conversion lives only inside the session-bound
  `_execute_set_pipeline` handler (`tools/sessions.py`), so reusing it here is
  costly and duplicates Task 3. Note full-state validation as the costlier
  alternative.
- [ ] Add narrow `.gitignore` negations for the new corpus directory. Because
  `.gitignore` L48 `/evals/*` excludes the directory, git cannot re-include
  files under it unless the directory itself is re-included first, in order:
  add `!/evals/composer-parity/` then `!/evals/composer-parity/**` immediately
  after L50 (mirroring the existing `!/evals/lib/` precedent) and **before** the
  generic `__pycache__` / `*.py[cod]` / `*.db` rules so junk stays ignored.
  Verify each committed fixture makes `git check-ignore -v <path>` **exit 1**
  (not ignored) and appear in `git ls-files`; and that
  `output/playwright/composer-parity/<x>` makes `git check-ignore` **exit 0**
  (already ignored by L379 — no `.gitignore` change needed there). Note
  `git check-ignore PATH` prints the path and exits 0 when it IS ignored — do
  not read that as "included".

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
  **Seam (do this — do not reuse the guided conftest):** monkeypatch the module
  global `elspeth.web.composer.service._litellm_acompletion`
  (`PlannerModelConfig.completion`, awaited at `pipeline_planner.py:1045`) to a
  scripted responder emitting the `emit_pipeline_proposal` terminal tool, and
  build a **real** `ComposerServiceImpl.for_trained_operator(...)` with
  `_compute_availability` monkeypatched available (per
  `tests/integration/web/composer/test_freeform_pipeline_planner.py`). Do NOT
  reuse `guided/conftest.py`'s `_DeterministicGuidedPlanner` — it constructs a
  `PipelineProposal` directly, which this task forbids. Configure a
  permissive-but-realistic web plugin policy that admits every plugin the 9
  fixtures use (the guided conftest deliberately restricts the allowlist to
  `transform:passthrough`).
  **Freeform recipe-bypass (false-green trap):** disable or guarantee non-match
  of the server recipe fast-path (`service.py:2534` `prepare_pipeline_plan`,
  before `plan_pipeline` at `:2567`) for the parity intent, so freeform
  provably traverses `plan_pipeline` and `build_planner_capability_manifest`
  rather than returning a recipe-router graph.
  **Guided-staged second completion binding:** the source/output/transform
  stage solvers call a *separate* by-name binding
  `elspeth.web.composer.guided.chat_solver._litellm_acompletion`
  (`chat_solver.py:68`) — patching the service global does NOT intercept it.
  Either patch `chat_solver._litellm_acompletion` too, or drive source+output
  stages deterministically via `/guided/respond` structured transitions and
  script only the topology (`plan_guided_pipeline`) completion.
- [ ] Traverse real prompt/tool assembly, terminal parser, custody, candidate,
  durable proposal, acceptance, audited `set_pipeline`, immutable state, and
  public YAML compiler. Do not inject a `PipelineProposal` or
  `CompositionState`.
- [ ] Normalize generated ids, connection names, temporary output paths,
  versions, timestamps, rationale, and profile metadata. Preserve plugin/node
  kinds, normalized options, edge roles/routes, topology, policies, merge mode,
  field contracts, output schemas, and failure policy. No graph-isomorphism /
  normalization helper exists anywhere under `tests/` today —
  `tests/helpers/composer_graphs.py` implements it **from scratch**; cite design
  §8.1 (`2026-07-13-composer-guided-freeform-capability-parity-design.md:677`)
  for the exact preserve-vs-canonicalize split. Lift the reusable deterministic completion
  double (`_ScriptedCompletion` + `_Response` / `_ToolCall` / `_pipeline` /
  `_response` helpers, `tests/unit/web/composer/test_pipeline_planner.py`; also
  mirrored in `test_freeform_pipeline_planner.py`) rather than re-inventing it —
  its stateful sequential form enables the one-repair and repair-exhaustion
  cases.
- [ ] Parameterize nine fixtures by three surfaces and assert graph isomorphism,
  equivalent validation, runtime graph, and public YAML semantics: 27 cases,
  no provider network, skips, or expected failures. The `structured_llm` (and
  colour two-LLM) fixtures cannot pass candidate validation until Task 1's typed
  queries and the §5.3 secret-reference probe fix land; gate those cases on
  Task 1.
- [ ] Add one-repair success, repair exhaustion, policy rejection, future-stage
  retention across restart, completed-stage back/edit, unavailable plugin, and
  tutorial identity cases. The **repair-exhaustion** and **policy-rejection**
  negatives on the freeform surface depend on **Task 0** (freeform
  `PipelinePlannerError` translation + closed failure-disposition record); do
  not assert a closed audit / "failed" disposition on freeform until Task 0
  lands.

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
- Modify: `pyproject.toml` only if the required marker/dependency is absent. In
  current reality Hypothesis is already a dependency and property tests need no
  custom marker, so this conditional resolves to a **no-op**; any
  determinism / example-count control belongs in the test's `@settings`, not
  `pyproject.toml`.

- [ ] Generate bounded policy-valid canonical DAGs with plural sources/outputs,
  transforms, gates, aggregation, queues, fork/coalesce, and error routes using
  deterministic Hypothesis settings — pin
  `@settings(max_examples=50, deadline=None, derandomize=True)` (or register a
  dedicated profile) so the count and determinism hold regardless of
  `HYPOTHESIS_PROFILE` (the loaded `ci` profile defaults to `max_examples=100`
  and is not derandomized). Reject invalid combinations in the strategy; every
  emitted case must pass the canonical boundary, reusing Task 2's
  `SetPipelineArgumentsModel` + policy loader for that check.
- [ ] Pass each generated case independently through all three real-path
  adapters (reuse Task 3's adapters and the
  `tests/helpers/composer_graphs.py` isomorphism helper — Task 4 cannot be built
  standalone) and compare committed runtime graphs by isomorphism. Shrinking
  must report intent, payload, surface, and first differing semantic attribute.
- [ ] Mutate the **single shared** advertised terminal schema as sent on ONE
  surface's adapter call. Post-plans-01-04 every surface uses one
  `canonical_set_pipeline_schema()` (`tools/schema_contract.py`); there is no
  per-surface "guided schema" to edit in isolation, so scope the
  `planner_terminal_tool_definition` (`pipeline_planner.py:330`) monkeypatch to
  that one adapter run. Apply: remove `fork_to`, remove `merge`, cap named
  sources at one, and — since `nodes[].node_type` is a **bare string**
  (`tools/sessions.py:1141`), not an enum — **introduce** a narrowing enum on
  `node_type` that omits `queue` (a genuine structural narrowing that changes
  the schema hash). Each control must trip the **schema-identity gate**
  `build_planner_capability_manifest` (`capability_skill.py:188`; the
  `stable_hash(advertised_schema) != stable_hash(canonical_schema)` check at
  `:223` raising `AuditIntegrityError`) — **NOT** the graph-isomorphism
  assertion, which a scripted fake completion would leave green because the
  committed graph is identical. The gate raises a generic message, so naming the
  field/surface is the **test's** responsibility: parametrize over
  `(surface, removed_field)`. A dictionary-only comparison does not count.

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
  provider responses. (On the freeform surface, the closed failure-disposition
  record this oracle can assert depends on **Task 0**.)
- [ ] Implement and **unit-test in `live_acceptance.py`** the evidence-directory
  hygiene / redaction logic (restrictive permissions, symlink rejection,
  resolved-revision-dir containment, and removal of credentials/cookies/raw
  provider responses). Task 6 only invokes this at runtime, so it is built once
  here — not duplicated in the runbook or dropped.
- [ ] Add negative fixtures for one LLM, missing branch row, wrong merge,
  unrouted failure, extra/missing field, JSONL root, duplicate identity,
  bool-as-number, NaN, nonterminal accounting, fake provider, excess repair,
  operator topology correction, and mixed revision.
- [ ] Verify `git ls-files --error-unmatch
  evals/composer-parity/live_acceptance.py` succeeds after creating the oracle;
  do not rely on a broad ignore exception that can silently omit it. This
  presupposes **Task 2** landed the `evals/composer-parity/` directory and its
  `.gitignore` negations; the new `tests/integration/evals/` package also needs
  a package `__init__` matching the `tests/unit/evals/` convention. Do not start
  Task 5 standalone.
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
  switch to freeform for a supported topology. When this work encodes
  schema/epoch numbers into the runbook, user-manual, or
  `test_composer_capability_docs.py`, use the **Plan 05** values (session epoch
  35, guided schema 10, landscape epoch 28) — the design doc §6.1's `8`/`28` are
  stale; do not encode them.

Run:

```bash
uv run pytest \
  tests/unit/evals/composer_parity/test_live_acceptance.py \
  tests/integration/evals/composer_parity/test_live_acceptance_server.py \
  tests/unit/docs/test_composer_capability_docs.py -q
cd src/elspeth/web/frontend
npm test -- --run src/components/chat/guided src/test/a11y/components.a11y.test.tsx
npm run typecheck
```

Expected: PASS, including every negative control.

## Task 6: Recreate staging, deploy, and run all three proofs

**Authoritative docs:**

- `docs/runbooks/staging-session-db-recreation.md` — specifically the
  "Staging Reset For `elspeth.foundryside.dev`" section (L363) and the
  "Skill Changes Require Service Restart, Not Reload" section (L349). This is
  the real staging shape: source-checkout systemd/Caddy, `elspeth-web.service`,
  deploy path `/home/john/elspeth`. Skill-pack prompt changes in this feature
  require `systemctl restart`, **not** reload.
- **Not** `docs/runbooks/aws-ecs-deployment.md` — AWS ECS Fargate
  (PostgreSQL/EFS/Cognito, immutable task defs) is a different topology and does
  not describe this staging target; `ansible-ubuntu-deployment.md` is
  self-declared spec-only.

- [ ] On the exact integrated revision, run the full backend, testcontainer,
  property, frontend, and Playwright gates. Assert session epoch 35, guided
  schema 10, landscape epoch 28, and no active chain proposal path. Cite
  `tests/integration/web/composer/guided/test_schema9_epoch.py` (stale-named; it
  asserts `SESSION_SCHEMA_EPOCH == 35` and `SQLITE_SCHEMA_EPOCH == 28`) for
  session/landscape, and `guided/state_machine.py:52`
  (`GUIDED_SESSION_SCHEMA_VERSION == 10`) for guided. **Re-grep**
  `SQLITE_SCHEMA_EPOCH` at the ACTUAL integrated/deploy revision after any
  rebase onto the release branch: worktree code confirms `28`, but if the
  integrated value is not 28 this becomes a **two-DB reset** (session +
  Landscape, one service-stop window) and both the runbook's Landscape-28 text
  and this assertion must be updated to the integrated value. (The Landscape MCP
  install reporting epoch 27/29 is that tool's own store, not the elspeth
  constant — do not change 28 on that basis.)
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
- [ ] Invoke the evidence-directory hygiene / redaction logic (restrictive
  permissions, symlink rejection, resolved-revision-dir containment, removal of
  credentials/cookies/raw provider responses before retention or cleanup). This
  logic is implemented and unit-tested in `live_acceptance.py` under **Task 5**;
  Task 6 only invokes it at runtime — it is built once, not twice or dropped.

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
# check_contracts is a known-RED baseline (elspeth-322b6c6f2a; ~26-34 pre-existing
# dict[str,Any]/type violations, operator ordered NO whitelisting). The Plan 05
# obligation is "introduce NO NEW findings under src/elspeth", NOT exit 0. Run it
# as a changed-files delta / no-new-findings check (or take the
# operator-sanctioned SKIP-with-reconciliation path the pubint batch used); do
# NOT demand exit 0 here.
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

Expected: every command exits 0 **except `check_contracts.py`** (see the note in
the gate block — its obligation is "no new `src/elspeth` findings", not exit 0),
and all three staging proofs pass on one revision.

**Definition of done:** The deterministic and generated suites detect any
surface narrowing, all three deployed ordinary authoring surfaces independently
derive and execute the colour graph, tutorial retains shared mechanical
identity, retained evidence is redacted and internally consistent, and the
controlling issue can close.
