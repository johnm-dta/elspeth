# DAG completeness evidence: authoring surfaces and contracts

**Assessment date:** 2026-07-15
**Repository snapshot:** `release/0.7.1`, HEAD `0dcd61aca`
**Scope:** freeform/runtime YAML, guided Composer authoring, import/export, runtime validation hand-off, documented execution-graph contract, and cross-surface acceptance evidence.

## Verdict

Authoring and contract completeness is **partial (2/5)**.

The runtime/freeform surface can represent the current graph vocabulary, and current source plus focused tests confirm that queues are imported and emitted. The principal incompleteness is no longer “queue syntax is absent”; it is that guided authoring cannot yet construct the same topology set, the planned cross-surface parity suite has not been delivered, and the normative execution-graph document still describes the older single-source, six-node model.

A topology should not be called authoring-complete until all supported authoring surfaces produce the same canonical graph, pass the same production validator/builder, round-trip without semantic loss, and have executable scenario evidence. On that standard, the product is materially short of complete even though the freeform/runtime path is capable.

## Assessment criteria

| Criterion | Complete means | Current assessment |
|---|---|---|
| Vocabulary coverage | Every runtime node/edge construct is expressible on every advertised surface | **Partial.** Freeform/runtime YAML covers the current vocabulary; guided mode does not. |
| Canonical equivalence | Equivalent guided and freeform inputs compile to the same canonical graph and execution plan | **Unknown.** A parity design exists, but its fixture/evaluation suite is not present. |
| Production validation | Authoring validation delegates to, or is proven equivalent to, the real runtime builder/validator | **Partial.** `validate_pipeline` is available as the runtime path; at least one web-versus-builder composition remains tracked as unconfirmed. |
| Semantic round-trip | import → edit/no-op → export → import preserves nodes, edges, labels, policies, and configuration semantics | **Partial.** Queue-focused importer/generator coverage passes; full-topology equivalence evidence is absent. |
| Failure UX | Invalid graphs fail before execution with the same actionable reason on each surface | **Unknown.** No complete cross-surface negative matrix was found. |
| Maintained contract | Normative docs and examples match the live node model, source cardinality, and authoring behavior | **Fail.** `docs/contracts/execution-graph.md` describes the superseded model. |
| Regression proof | Each mandatory topology is tested through all authoring surfaces and the production build path | **Fail.** The planned 9-topology × 3-surface matrix is not implemented. |

## Confirmed current capability

### Runtime and freeform YAML

- `NodeType` includes `QUEUE` alongside source, transform, gate, aggregation, coalesce, and sink (`src/elspeth/contracts/enums.py`, `NodeType`, lines 91–103 in the current source view).
- `CompositionState` is the public Composer model and has its own validation path (`src/elspeth/web/composer/state.py`, `CompositionState` and `CompositionState.validate`).
- Runtime YAML import explicitly parses queue mappings through `_queues_from_runtime_mapping`, then assembles them in `composition_state_from_runtime_yaml` (`src/elspeth/web/composer/yaml_importer.py`).
- Queue emission is present in `src/elspeth/web/composer/yaml_generator.py`; focused importer/generator tests pass.
- Web execution has access to the production validation path through `src/elspeth/web/execution/validation.py::validate_pipeline`. This is the correct convergence point for proving authoring/runtime equivalence; the evidence does not yet show that every Composer validation path is mechanically identical to it.

### Focused test evidence

The parent assessment run executed:

```text
uv run pytest tests/unit/web/composer/test_yaml_importer.py \
  tests/unit/web/composer/test_yaml_generator.py -q
66 passed in 6.14s
```

Queue-specific evidence includes tests for an empty queue, a queue with a description, a multi-source queue example, and YAML round-trip behavior in `tests/unit/web/composer/test_yaml_importer.py`.

This proves the narrow queue import/export claim. It does **not** prove guided parity, canonical graph equivalence, production execution, failure behavior, or round-trip equivalence for the full topology vocabulary.

## Material gaps

### AC-01 — Guided mode is not topology-complete (P1)

**Status:** confirmed gap.
**Surface:** `src/elspeth/web/composer/`; guided Composer UX/state.
**Evidence:** the approved design in `docs/superpowers/plans/composer-parity/2026-07-13-composer-guided-freeform-capability-parity-design.md` records that guided mode currently authors a linear transform list rather than the full graph. It cannot yet author multiple sources, arbitrary routes, forks, queues, coalesces, aggregations, gates, explicit edges, or multiple outputs at freeform parity.

**Why it matters:** users can build materially different systems depending on which supported authoring surface they choose. Runtime capability therefore overstates product capability.

**Exit evidence:** construct every mandatory topology in guided mode, compile it through the production builder, and show semantic equivalence to the corresponding freeform fixture.

### AC-02 — The cross-surface parity matrix is planned, not executable (P1)

**Status:** confirmed evidence gap.
**Surface:** Composer evaluation and integration tests.
**Evidence:** `docs/superpowers/plans/composer-parity/2026-07-17-composer-capability-parity-plan-05-verification-acceptance.md` specifies nine fixtures—linear transform, conditional gate, multi-output, fork/coalesce, multi-source/queue, aggregation, row expansion, error routing, and structured LLM—across three authoring surfaces, for 27 cases. The planned `evals/composer-parity/` and `tests/integration/web/composer/parity/` suites were absent in the live repository inspection.

**Why it matters:** unit coverage of import/export helpers cannot detect semantic drift between surfaces or between the web model and execution builder.

**Exit evidence:** 27 passing positive cases plus negative cases for invalid routes, fan-in without queue/coalesce, missing required destinations, incompatible schemas, and unsupported node/plugin combinations.

### AC-03 — Composer acceptance may diverge from the real builder (P1, unconfirmed defect)

**Status:** triage; likely stale or already mitigated, but not regression-proven.
**Tracker:** `elspeth-a6ca0bef77`.
**Evidence:** the issue reports that web validation accepts fork → coalesce → fork while the real builder may fail because the intermediate coalesce has no `output_schema_config`. Current builder code now has a `deferred_config_gate_schemas` pass that resolves coalesce-producer schemas (`src/elspeth/core/dag/builder.py`, lines 1081–1085), so the original failure premise may no longer hold.

**Why it matters:** authoring success followed by build-time failure is a contract breach, and the topology is central to claims of compositional closure.

**Exit evidence:** a single integration fixture must pass Composer validation, `validate_pipeline`, `build_execution_graph`, and a minimal execution for fork → coalesce → fork. If it fails, fix the schema propagation boundary rather than weakening validation.

### AC-04 — The normative execution-graph contract is stale (P1)

**Status:** confirmed documentation defect.
**Surface:** `docs/contracts/execution-graph.md`.
**Evidence:** the document says there are six node types, requires exactly one source, presents a singular-source facade, and omits queues. Live code has seven node types and supports plural sources with explicit queue fan-in.

**Why it matters:** this is not cosmetic drift. It misstates legal graph structure and would cause reviewers, plugin authors, and users to reject supported graphs or design against the wrong invariants.

**Exit evidence:** update the contract only after the capability matrix is executable; bind each structural claim to a source invariant and at least one test.

### AC-05 — Round-trip proof is narrower than the advertised graph model (P2)

**Status:** confirmed evidence gap.
**Surface:** YAML importer/generator and Composer persistence/export.
**Evidence:** queue-focused and general helper tests pass, but there is no demonstrated all-topology invariant that import/export preserves graph semantics, including route labels, coalesce policies, aggregation settings, error destinations, schema declarations, and plugin configuration.

**Exit evidence:** compare canonicalized graph representations—not raw YAML text—after freeform import/export and guided edit/export for every parity fixture.

### AC-06 — Browser-level acceptance remains skipped or pending (P2)

**Status:** confirmed evidence gap.
**Tracker:** `elspeth-7cf763da7c`.
**Evidence:** topology, mandatory-field, YAML export round-trip, and schema-preview parity Playwright specifications are skipped/pending.

**Why it matters:** Python model tests do not cover client-side state loss, hidden mandatory fields, invalid edge interactions, or export behavior in the actual user flow.

**Exit evidence:** enable the specifications against deterministic fixtures and make them required in the relevant CI lane.

### AC-07 — Tracker descriptions lag live queue support (P2 process gap)

**Status:** confirmed stale issue framing.
**Trackers:** `elspeth-a5b86149d4`, `elspeth-6421ffa028`.
**Evidence:** live `NodeType`, `CompositionState`, YAML importer/generator code, and 66 passing focused tests show queue representation and YAML import/export support. Any issue text that still claims queues cannot be represented or imported is no longer an accurate statement of the current code.

**Required action:** re-test each issue’s original reproducer, then close it if satisfied or rewrite it around the remaining narrow defect. In particular, `elspeth-a5b86149d4` retains a legitimate row-union/merge need even though its queue-exposure premise is stale. Do not equate pass-through queue fan-in with row-union semantics, and do not use stale issue prose as evidence that queue authoring is absent.

### AC-08 — Graph configuration contracts expose sensitive raw values (P2 tracker priority; completeness hard gate)

**Status:** confirmed cross-cutting contract gap.
**Trackers:** `elspeth-69c957ed96`, `elspeth-c4080bfb06`.
**Evidence:** topology hashing consumes raw node configuration, and DAG metadata retains raw plugin configuration. Authoring/import may therefore feed secret-bearing values into persisted graph identity or metadata surfaces.

**Why it matters:** lossless round-trip and canonical equivalence must not mean copying secret material into hashes, audit metadata, exports, or diagnostics. Secret exposure is a hard blocker for a production-complete authoring contract even if the issues are currently prioritized P2.

**Exit evidence:** define a canonical redacted/fingerprinted configuration representation, prove stable graph identity for equivalent secret references, and test that persisted metadata, exported YAML, errors, and audit events do not contain secret values.

## Scenario matrix

Legend: **P** proven by current focused evidence; **△** partial; **F** known missing; **U** unknown/unproven.

| Scenario | Freeform/import | Guided authoring | Production validation/build | Semantic round-trip | Browser E2E |
|---|---:|---:|---:|---:|---:|
| Linear transform | P | P | P | △ | U |
| Conditional gate/routes | △ | F | △ | U | U |
| Multiple outputs | △ | F | △ | U | U |
| Fork → coalesce | △ | F | △ | U | U |
| Fork → coalesce → fork | △ | F | U | U | U |
| Multiple sources → queue | P | F | △ | △ | U |
| Aggregation | △ | F | △ | U | U |
| Row expansion | △ | F | △ | U | U |
| Error/quarantine routing | △ | F | △ | U | U |
| Structured LLM/plugin configuration | △ | F | △ | U | U |

“Partial” in the freeform column means the syntax/model exists but this assessment did not find full cross-layer scenario proof. Unknown cells are completeness gaps, not presumed passes.

## Recommended shore-up sequence

1. **Make one canonical capability manifest.** Enumerate node types, legal edge forms, source/sink cardinality, route-label requirements, fan-in rules, schema behavior, and error destinations. Derive documentation and parity fixtures from this manifest or test it directly.
2. **Close the validation boundary first.** Reproduce `elspeth-a6ca0bef77` and require authoring validation to call the production validation/build path, or prove strict equivalence with differential tests.
3. **Deliver guided graph authoring.** Implement full node/edge operations and mandatory configuration for the same constructs accepted by freeform YAML.
4. **Implement the 27-case parity suite.** Assert canonical graph equality, production build success, diagnostics for invalid variants, and semantic round-trip. Add scale variants only after correctness is stable.
5. **Enable browser acceptance.** Remove the relevant Playwright skips and gate merges on topology creation, required fields, schema preview, and export/import flows.
6. **Repair the maintained contract.** Update `docs/contracts/execution-graph.md` and examples to the plural-source, seven-node model, with links to the executable scenarios.
7. **Harden configuration contracts.** Canonicalize or fingerprint secret references and prevent raw plugin configuration from entering graph metadata, exports, hashes, errors, or audit evidence.
8. **Reconcile Filigree.** Close or narrow the two queue-era issues only after replaying their original reproducers; preserve the row-union need in `elspeth-a5b86149d4`, and keep guided parity and browser work open until their acceptance evidence exists.

## Completion gate for this domain

Authoring/contracts may be scored complete only when:

- all mandatory topologies are constructible in freeform and guided modes;
- equivalent inputs compile to an identical canonical graph and production traversal plan;
- production validation rejects the same invalid graphs with surface-appropriate but semantically equivalent diagnostics;
- import/export is semantically lossless across every fixture;
- browser-level authoring and export flows are active CI checks;
- the execution-graph contract matches live invariants; and
- graph identity, metadata, exports, and diagnostics cannot reveal raw secrets; and
- no mandatory matrix cell is unknown, skipped, or plan-only.

## Evidence limitations

Loomweave entity evidence, live Filigree reconciliation, and the focused test result above were gathered during this assessment. The shared shell later became unavailable because the host exhausted file descriptors, so this sub-analysis could not independently rerun the parent-run tests after that interruption. Tracker conclusions deliberately distinguish live-code contradictions from issue status: queue-support claims are source/test-confirmed, while tracker closure still requires replaying each issue’s reproducer through the normal Filigree workflow.
