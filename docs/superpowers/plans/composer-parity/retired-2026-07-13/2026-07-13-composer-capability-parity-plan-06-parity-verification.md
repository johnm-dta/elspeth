# Composer Capability Parity Plan 06: Parity Verification Implementation Plan

> **RETIRED (2026-07-17): DO NOT EXECUTE.** See
> [the current disposition](../2026-07-17-current-plan-disposition.md).

**Goal:** Prove all authoring surfaces can independently produce every supported pipeline class, and make any future guided capability reduction fail deterministically.

**Architecture:** A versioned fixture corpus drives the real prompt/response parser, shared planner, immutable proposal, unpublished candidate, audited `set_pipeline` commit, and runtime YAML compiler. Graph isomorphism establishes semantic equivalence; Hypothesis explores valid DAGs; schema mutation and profile-identity controls prove the suite can detect drift.

**Tech Stack:** pytest, Hypothesis, NetworkX `MultiDiGraph`, fake completion provider, composer audit store, runtime YAML loader.

---

## File structure

**Create:**
- `evals/composer-parity/README.md`
- `evals/composer-parity/fixtures/*.json`
- `evals/composer-parity/fixtures/two_llm_colour.csv`
- `evals/composer-parity/fixtures/two_llm_colour_request.txt`
- `tests/integration/web/composer/parity/conftest.py`
- `tests/integration/web/composer/parity/test_fixture_matrix.py`
- `tests/integration/web/composer/parity/test_schema_mutation_controls.py`
- `tests/integration/web/composer/parity/test_guided_deferral.py`
- `tests/integration/web/composer/parity/test_repair_policy.py`
- `tests/integration/web/composer/parity/test_tutorial_identity.py`
- `tests/property/web/composer/test_authoring_surface_parity.py`
- `tests/helpers/composer_graphs.py`

**Modify:**
- `pyproject.toml` — register markers/dependencies only if not already present.
- `src/elspeth/web/composer/guided/prompts.py` — expose effective manifest to tests.

### Task 1: Build the nine-class canonical fixture corpus

**Files:**
- Create: `evals/composer-parity/README.md`
- Create: `evals/composer-parity/fixtures/*.json`
- Create: `evals/composer-parity/fixtures/two_llm_colour.csv`
- Create: `evals/composer-parity/fixtures/two_llm_colour_request.txt`

- [ ] **Step 1: Add fixture schema and provenance rules**

Each fixture stores the plain-English intent, canonical accepted arguments,
expected node/edge semantic attributes, runtime assertions, and stable fixture
version. It must not contain a scripted sequence of composer tool calls.

- [ ] **Step 2: Encode all nine topology classes**

Create these fixtures:

1. `linear_transform`: CSV raw -> passthrough -> JSON clean.
2. `conditional_gate`: fixed `amount:int` -> `amount >= 50` -> accepted/rejected.
3. `multi_output`: distinct named outputs and write-failure policies.
4. `fork_coalesce`: true gate, `fork`/`fork_to`, branch transforms, `require_all` coalesce.
5. `multi_source_queue`: orders/refunds publish `inbound`, queue consumes it, downstream transform.
6. `aggregation`: `batch_stats` count trigger, aggregate mode, required fields.
7. `row_expansion`: source guarantees `color_name`, `hex`, `copies`; `batch_replicate` preserves them and adds `copy_index` for a downstream consumer.
8. `error_routing`: source validation failure plus transform `on_error` routes to a failure sink.
9. `structured_llm`: typed structured fields from one named LLM query are consumed downstream through a secret reference.

- [ ] **Step 3: Pin the exact live colour inputs**

The ten-row CSV must hash to:

```text
067f0ffeb6a349fc33c1ce2f65cac65dcb37eb3bdd30ef8ca2670439238ba702
```

Write `two_llm_colour_request.txt` as UTF-8 with exactly this single line and a
terminal LF:

```text
Create and run a pipeline over the uploaded ten-row colour CSV. For every row, send color_name and hex to two independent LLMs in parallel. One LLM must assess the amount of blue as an integer from 0 to 100, a confidence from 0 to 1, and a concise non-empty reason. The other must independently assess the amount of red using the same types and ranges. Wait for both assessments and merge them back into one hybrid row. Keep exactly color_name, hex, blue_amount, blue_confidence, blue_reason, red_amount, red_confidence, and red_reason. Remove raw responses, token or model metadata, and branch bookkeeping. Write successful rows as one JSON array and failures to a separate JSON output. Use deterministic sampling where supported and an economical configured model.
```

Its SHA-256 must be:

```text
37562b0fcfad56182dd33b3b72457681959ffa71f159beabcd387170187987d2
```

The request deliberately names no composer tool or plugin. It states outcomes
and lets the model derive plugin discovery, configuration, and call order.

- [ ] **Step 4: Validate fixtures and commit**

```bash
sha256sum evals/composer-parity/fixtures/two_llm_colour.csv
sha256sum evals/composer-parity/fixtures/two_llm_colour_request.txt
uv run python -m json.tool evals/composer-parity/fixtures/linear_transform.json >/dev/null
git add evals/composer-parity
git commit -m "test(composer): add canonical parity fixture corpus"
```

### Task 2: Build a real-path deterministic authoring harness

**Files:**
- Create: `tests/integration/web/composer/parity/conftest.py`
- Create: `tests/helpers/composer_graphs.py`
- Test: `tests/integration/web/composer/parity/test_fixture_matrix.py`

- [ ] **Step 1: Write one failing end-to-end harness case**

Use a deterministic fake completion with three distinct production adapters:

- freeform calls the normal compose route/`ComposerServiceImpl.compose()`;
- guided-full calls the authenticated server-owned full-guided evaluation seam;
- guided-staged calls `/guided/start` and progresses every persisted stage to
  wire review.

Each traverses the actual response schema, parser, `plan_pipeline()`, proposal
checkpoint, candidate validation, audited `set_pipeline` commit, and runtime
YAML loader. A direct planner adapter, fake proposal, or mocked executor is not
acceptable except inside the explicitly authenticated guided-full server seam.
For staged, assert the route transcript, persisted transitions, reviewed facts,
planner invocation, proposal review, and commit.

- [ ] **Step 2: Implement semantic graph normalization**

Represent committed pipelines as `networkx.MultiDiGraph`. Node matching must
include plugin, node type, normalized options, required/guaranteed fields,
trigger/output mode, and output binding. Edge matching must include source
output, target input, route/fork metadata, merge strategy, and error routing.

- [ ] **Step 3: Assert custody and audit invariants**

For every run, assert the proposal content hash matches accepted exact
arguments, invalid candidates never become current, the accepted audit record
exists, secrets remain references, and runtime YAML compiles the committed
state.

- [ ] **Step 4: Run and commit**

```bash
uv run pytest tests/integration/web/composer/parity/test_fixture_matrix.py -k linear_transform -q
git add tests/integration/web/composer/parity/conftest.py tests/integration/web/composer/parity/test_fixture_matrix.py tests/helpers/composer_graphs.py
git commit -m "test(composer): exercise the real canonical planning path"
```

### Task 3: Run the 27-case deterministic surface matrix

**Files:**
- Modify: `tests/integration/web/composer/parity/test_fixture_matrix.py`

- [ ] **Step 1: Parameterize nine fixtures across three authoring surfaces**

```python
@pytest.mark.parametrize("surface", ["freeform_big_bang", "guided_full", "guided_staged"])
@pytest.mark.parametrize("fixture_name", CANONICAL_FIXTURES)
async def test_surface_authors_graph_isomorphic_pipeline(surface, fixture_name, parity_harness): ...
```

Every case starts from the same English intent and discovered capabilities. Do
not let guided reuse a freeform proposal or accepted pipeline. Reuse the three
production adapters from Task 2 for both deterministic and property cases.

- [ ] **Step 2: Add runtime-specific assertions**

Execute or compile the fixture far enough to prove queue, aggregation,
replication, structured field consumption, and error routes retain their
semantics. Explicitly assert `batch_replicate` propagates upstream guaranteed
fields plus `copy_index`.

- [ ] **Step 3: Record per-surface manifests**

Persist test artifacts containing fixture id, surface, proposal hash, graph
digest, effective schema hash, core hash, and audit event id. Never persist raw
secrets or full provider responses.

- [ ] **Step 4: Run and commit**

```bash
uv run pytest tests/integration/web/composer/parity/test_fixture_matrix.py -q
git add tests/integration/web/composer/parity/test_fixture_matrix.py
git commit -m "test(composer): enforce the 27-case parity matrix"
```

### Task 4: Add stage-deferral and repair-policy regressions

**Files:**
- Create: `tests/integration/web/composer/parity/test_guided_deferral.py`
- Create: `tests/integration/web/composer/parity/test_repair_policy.py`

- [ ] **Step 1: Test deferred LLM intent survives restart**

At source review, ask for an LLM transform. Assert an explicit target-stage
deferral, no stage advance, no repair count, no plugin configuration, and no
capability disclaimer. Reload the session, reach Step 3, and assert the LLM
intent is injected and consumed only after a reviewed fact/proposal covers it.

- [ ] **Step 2: Cover other directions and unavailable plugins**

Request a sink during source review and topology during sink review. Assert
ordered persistence and correct consumption. Request a nonexistent plugin and
assert a catalog availability error with no deferral. At transformation review,
request a different source and assert stable-id back/edit with no deferred
record targeting the already completed source stage.

- [ ] **Step 3: Pin deterministic repair behavior**

Feed recoverable invalid planner output and assert the same bounded repair path
on all three surfaces. Feed a policy-rejected or still-invalid candidate and
assert no commit, no current-state replacement, and no freeform handoff.

- [ ] **Step 4: Run and commit**

```bash
uv run pytest tests/integration/web/composer/parity/test_guided_deferral.py tests/integration/web/composer/parity/test_repair_policy.py -q
git add tests/integration/web/composer/parity/test_guided_deferral.py tests/integration/web/composer/parity/test_repair_policy.py
git commit -m "test(guided): preserve wrong-stage intent and repair policy"
```

### Task 5: Add generated-DAG parity and schema mutation controls

**Files:**
- Create: `tests/property/web/composer/test_authoring_surface_parity.py`
- Create: `tests/integration/web/composer/parity/test_schema_mutation_controls.py`
- Modify: `pyproject.toml`

- [ ] **Step 1: Generate policy-valid canonical DAGs**

Generate bounded plural sources/outputs, transform/aggregation/queue nodes,
fork/coalesce/merge combinations, and error routes. Use:

```python
@settings(max_examples=50, derandomize=True, deadline=None)
```

Reject examples only in the generator; all emitted payloads must pass the real
shared web policy before surface comparison.

- [ ] **Step 2: Compare all surfaces by graph isomorphism**

Each surface independently authors the generated case through the real-path
harness. Shrinking must report the English intent, canonical payload, surface,
and first differing node/edge attribute.

- [ ] **Step 3: Prove the real parity assertion catches advertised-schema drift**

Run controlled in-memory schema mutations and require failures for:

1. deleting `fork_to`;
2. injecting a narrowing `node_type` enum that excludes `queue` into one
   surface (the current canonical field is a plain string);
3. deleting `merge`;
4. setting named sources to `maxProperties: 1`.

For each mutation, first run the unmodified three-surface case and require it to
pass. Then inject the mutation into exactly one surface's actual terminal schema
immediately before the wrapped completion call, invoke the same real-path parity
assertion, and require a typed `SurfaceSchemaMismatch` naming the mutated field,
the affected surface, and the untouched reference surfaces. A test that only
compares schema dictionaries or asserts the mutation exists does not count.

- [ ] **Step 4: Run and commit**

```bash
uv run pytest tests/property/web/composer/test_authoring_surface_parity.py tests/integration/web/composer/parity/test_schema_mutation_controls.py -q
git add tests/property/web/composer/test_authoring_surface_parity.py tests/integration/web/composer/parity/test_schema_mutation_controls.py pyproject.toml
git commit -m "test(composer): property-check authoring surface parity"
```

### Task 6: Guard tutorial-profile identity

**Files:**
- Create: `tests/integration/web/composer/parity/test_tutorial_identity.py`
- Modify: `src/elspeth/web/composer/guided/prompts.py`

- [ ] **Step 1: Expose the effective-call manifest**

Include planner implementation id, actual rendered prompt hash, shared-core
hash, canonical schema hash, effective tool-schema hash, surface, and profile.

- [ ] **Step 2: Assert tutorial and staged identity**

Hash the exact messages and tools received by the wrapped completion call and
compare them to the manifest and audit record; self-reported helper hashes are
not sufficient. Tutorial may have a distinct rendered-prompt hash, but it must have the same
planner implementation, shared core, canonical schema, and effective tools as
guided-staged. Run both the fixed tutorial lesson and a non-tutorial complex
fixture through the real tutorial staged protocol.

- [ ] **Step 3: Run the Plan 06 gate and commit**

```bash
uv run pytest tests/integration/web/composer/parity tests/property/web/composer/test_authoring_surface_parity.py -q
git add src/elspeth/web/composer/guided/prompts.py tests/integration/web/composer/parity/test_tutorial_identity.py
git commit -m "test(guided): prevent tutorial capability regression"
```

## Plan 06 completion gate

- [ ] All 27 deterministic surface/fixture cases pass independently.
- [ ] Fifty deterministic generated valid DAGs pass graph-isomorphism parity.
- [ ] All four schema mutations are detected by the suite.
- [ ] Wrong-stage requests are deferred, persisted, injected, and consumed correctly.
- [ ] Tutorial-profile shares planner/schema/tools/core identity with guided-staged.
- [ ] Evidence proves proposal custody, audit, secret-reference, and runtime compilation invariants.
