# Maintained DAG Scenario Corpus Design

**Date:** 2026-07-17
**Status:** Approved design, pre-implementation
**Branch context:** `codex/dag-scenario-corpus` from `release/0.7.1`
**Filigree:** `elspeth-ef29ef6ba4`

## Purpose

Create one maintained, table-driven acceptance spine for the fifteen mandatory
DAG scenarios defined by
`docs/architecture/dag/completeness-criteria.md`. The corpus must distinguish
real production-path proof from direct graph construction, unit-only evidence,
plans, skips, and unknowns. It must also provide a reusable harness that drives
one authored configuration through validated settings, production plugin
instantiation, canonical graph construction, runtime execution, durable audit,
database reopen, and public recovery.

This package establishes the corpus contract and the common execution/evidence
machinery. It does not claim that all fifteen scenarios are already complete.
Current gaps remain visible, owned, and measurable until later remediation
promotes their cells to passing executable evidence.

## Context

The permanent DAG information area already defines:

- a strict authoring-to-CI completeness chain;
- fifteen mandatory scenarios;
- eleven scenario-matrix dimensions;
- `U` as an evidence result rather than implied success; and
- a hard gate while a mandatory scenario contains unknown, skipped, or
  plan-only evidence.

The current assessment records that no scenario is complete across every
applicable dimension. Existing evidence is distributed across production-path
integration tests, direct graph tests, recovery harnesses, browser tests, and
dated reports. Without one machine-checked corpus, those surfaces can silently
describe different graphs or leave gaps invisible to CI.

## Decisions

1. A versioned YAML manifest in the permanent DAG information area is the
   authoritative scenario and evidence inventory.
2. The manifest contains exactly the fifteen stable scenario IDs and the closed
   scenario-dimension set. Set equality, not mere minimum coverage, is enforced.
3. Strict frozen Pydantic models in the test support layer define the common
   schema. Unknown fields, coercion-dependent values, duplicate identifiers,
   and inconsistent status/evidence combinations fail closed.
4. Scenario input fixtures live in a versioned shared test-fixture directory
   and are referenced by the manifest. Runtime, recovery, authoring, and browser
   tests must reuse those fixtures rather than recreate equivalent-looking
   configurations independently.
5. A passing cell requires executable evidence. Direct graph construction,
   documentary claims, or a historical assessment cannot by themselves make a
   production-path cell pass.
6. Current gaps are represented as `partial`, `fail`, or `unknown`, never
   `skip` or `xfail`. Every actionable non-pass cell carries a reason, a
   Filigree owner, and an observable exit gate; `not_applicable` instead
   requires a narrow applicability reason.
7. CI validates the complete manifest and executes every declared production
   harness case. It remains green while gaps are honestly represented, but the
   derived corpus verdict must remain `not_complete` until every applicable
   mandatory cell passes.
8. The common harness uses public production boundaries and file-backed durable
   stores. It does not use `LandscapeDB.in_memory()`, hand-assembled
   `PipelineConfig`, or direct-only graph fixtures as end-to-end proof.
9. Recovery proof closes and reopens the same database, retains the same
   payload directory, rebuilds fresh plugins and DB-bound services, and enters
   through `RecoveryManager.get_resume_point()` and `Orchestrator.resume()`.
10. Portable audit export and direct durable-state assertions are complementary.
    Checkpoint identity and coordination markers are captured directly because
    `LandscapeExporter` deliberately does not export those tables.

## Alternatives Considered

### Python-only scenario definitions

Python dataclasses or pytest parameter tables would be easy to implement, but
they would be less reviewable as an evergreen architecture contract and harder
for browser, authoring, and non-runtime tooling to consume. Rejected as the
authoritative representation.

### Existing-test inventory only

A manifest containing only pytest node IDs would make current evidence easier
to find, but it would not ensure that the tests use the same configuration or
exercise the complete production chain. Retained only as one evidence-reference
kind; rejected as the corpus architecture.

### Versioned YAML plus strict typed loader

Selected. YAML matches the product configuration and repository CI conventions,
while strict models provide closed vocabularies and cross-field validation.

## Repository Layout

```text
docs/architecture/dag/
└── scenario-corpus/
    ├── README.md
    └── v1/
        └── manifest.yaml

tests/fixtures/dag_scenario_corpus/
├── __init__.py
├── schema.py
├── loader.py
├── harness.py
├── plugins.py
└── v1/
    └── <scenario-id>/
        └── <variant>.yaml

tests/unit/architecture/
└── test_dag_scenario_corpus_contract.py

tests/integration/core/dag/
└── test_dag_scenario_production_path.py
```

The documentation README explains how to add evidence without rewriting dated
assessments. The root DAG README links the live corpus alongside the stable
criteria and current assessment.

## Manifest Contract

### Stable scenario IDs

The closed v1 set is:

1. `linear`
2. `multiple-independent-sources`
3. `multi-source-queue-fan-in`
4. `conditional-routing`
5. `fork-multiple-terminals-partial-failure`
6. `fork-coalesce-policies`
7. `sequential-nested-fork-coalesce`
8. `parallel-coalesces`
9. `aggregation-immutable-batch`
10. `row-expansion-parent-child-recovery`
11. `row-union-interleave`
12. `retry-quarantine-discard-routed-errors`
13. `sink-write-pending-redrive`
14. `checkpoint-deterministic-resume`
15. `multi-worker-lease-reclaim-late-completion`

Each record also carries the normative ordinal and title so accidental reorder,
rename, omission, or duplication is detected.

### Closed dimensions

Each scenario contains exactly these cells:

- `config`
- `build`
- `contracts`
- `runtime`
- `audit`
- `recovery`
- `concurrency`
- `freeform`
- `guided`
- `round_trip`
- `scale`

Security remains a cross-cutting hard gate, matching the current assessment,
rather than an additional per-scenario matrix column.

### Cell statuses and invariants

The status vocabulary is:

- `pass`
- `partial`
- `fail`
- `unknown`
- `not_applicable`

Cross-field rules are fail closed:

- `pass` requires at least one executable evidence reference;
- `partial`, `fail`, and `unknown` require `reason`, `owner_issue`, and
  `exit_gate`;
- `not_applicable` requires a narrow applicability reason and cannot carry an
  owner that implies deferred work;
- an executable evidence reference names a registered harness case or an exact
  pytest node ID;
- fixture references remain inside the versioned fixture root;
- all issue IDs use the `elspeth-` form; and
- the derived verdict is `not_complete` whenever any applicable cell is not
  `pass`.

The manifest records declared truth; it does not contain a manually editable
aggregate score or verdict.

## Common Evidence Schema

Two related models share the same closed stage vocabulary.

### Declared evidence

`EvidenceReference` identifies what supports one manifest cell:

- stable evidence ID;
- evidence kind (`harness`, `pytest`, `document`, or `decision`);
- exact locator;
- claim proved by that evidence; and
- production-path stages actually exercised.

The schema prevents a direct graph test from claiming config/runtime/recovery
coverage by requiring the exercised stages to be explicit.

### Observed run evidence

The harness returns an immutable `ScenarioRunEvidence` rather than ad hoc
tuples. It contains:

- corpus version, scenario ID, variant ID, and fixture digest;
- config parse/validation result;
- canonical graph summary and topology identity;
- build and validation result or stable rejection information;
- runtime `RunResult` summary and durable output observations;
- audit record counts and selected exact row/token/outcome/routing/scheduler
  facts;
- recovery precondition, checkpoint identity, reopen proof, resume result, and
  post-resume durable state; and
- the stages successfully completed.

Secrets and raw credential-bearing configuration are excluded. Evidence uses
canonical digests and redacted/config-safe facts already owned by production
boundaries.

## Production-Path Harness

### Build path

For each executable variant the harness:

1. reads the shared YAML fixture;
2. renders only harness-owned temporary paths and deterministic row data;
3. calls `load_settings_from_yaml_string()`;
4. calls `instantiate_plugins_from_config(..., preflight_mode=True)`;
5. projects delayed audit-export sinks with `execution_sinks_for_runtime()`;
6. calls `ExecutionGraph.from_plugin_instances()` with sources, source settings,
   transforms, projected sinks, aggregations, gates, queues, and coalesces;
7. calls structural and edge-compatibility validation; and
8. calls `assemble_and_validate_pipeline_config()` using the same plugin
   instances and graph.

Every build creates a fresh plugin bundle. Aggregation assembly mutates plugin
node identity and runtime closes plugin lifecycles, so instances are never
shared between oracle, interrupted, or resumed executions.

### Runtime and audit path

The harness creates a file-backed `LandscapeDB` and
`FilesystemPayloadStore`, enables checkpointing for recovery variants, and
calls `Orchestrator.run(..., settings=settings)`. It verifies the declared
`RunResult`, durable sink artifact, and Landscape facts before deriving the
portable terminal record set through `LandscapeExporter.export_run()`.

The initial executable happy-path fixture uses built-in CSV, passthrough, and
JSON plugins. This proves real discovery, configuration, graph construction,
runtime lifecycle, filesystem effect, and audit export rather than a test-only
pipeline assembly.

### Recovery path

Recovery variants use a dedicated config-instantiable fixture source whose
test controller triggers a deterministic shutdown at a declared durable seam.
The fixture plugin is discovered through the normal runtime factory; only the
plugin registry is replaced with a deterministic test registry.

The recovery sequence is:

1. execute the real pipeline to an interrupted checkpoint;
2. assert the failed/interrupted run state, nonterminal work where applicable,
   and latest checkpoint identity;
3. close the original `LandscapeDB`;
4. reopen the same database with
   `LandscapeDB.from_url(..., create_tables=False)`;
5. recreate the payload store, checkpoint manager, recovery manager,
   orchestrator, and all plugins;
6. rebuild an equivalent validated graph/config from the same fixture;
7. verify `RecoveryManager.can_resume()` and obtain the opaque resume point;
8. call `Orchestrator.resume()`; and
9. assert terminal work, exact outcomes/effects, no source replay, checkpoint
   cleanup, and terminal audit export.

No correctness assertion depends on sleeps. A test clock or deterministic
plugin control owns any time/interrupt transition.

## Initial Executable Coverage

This implementation establishes two executable v1 cases:

- a linear happy path covering config, production build/validation, runtime,
  durable JSON output, and audit export; and
- a deterministic checkpoint/reopen/resume case covering the complete
  config-to-recovery chain with fresh-process-equivalent objects.

All fifteen scenarios are present from the first manifest commit. Cells not yet
proved by these cases retain the current assessment's honest status, issue
owner, and exit gate. Later remediation extends the same manifest and fixture
registry rather than adding parallel acceptance suites.

## CI Contract

Ordinary unmarked unit tests enforce:

- manifest parse and strict schema validation;
- exact scenario and dimension set equality;
- stable ordinal/title mapping;
- status/evidence/owner/exit-gate invariants;
- fixture and evidence-locator integrity;
- harness-case/manifest registration parity; and
- derived `not_complete` truth while gaps remain.

Ordinary integration tests parameterize the executable harness cases directly
from the manifest. Missing fixtures or registered cases fail collection/test
execution; they do not skip. No bespoke CI script is added because normal test
lanes already provide the required blocking gate on supported Python versions.

Promotion of a cell to `pass` requires adding the executable evidence and
changing the manifest in the same commit. Removing a registered case while its
evidence remains referenced fails parity checks.

## Error Handling

- Manifest errors name the scenario, dimension, offending field, and closed
  alternatives.
- Production build rejection is evidence only when the scenario declares an
  unsupported/rejection variant with an exact expected diagnostic class and
  stable predicate.
- Unexpected plugin, runtime, export, reopen, or resume failures propagate and
  fail the integration row.
- Audit corruption, topology mismatch, payload loss, or an ambiguous external
  sink effect fails closed; the harness never converts these into partial
  success.

## Documentation and Ownership

`docs/architecture/dag/scenario-corpus/README.md` documents:

- the manifest/schema relationship;
- how to add or promote evidence;
- why unknowns are not skipped;
- how current live truth differs from immutable dated assessments; and
- how later authoring, contention, and scale tasks extend the corpus.

The live manifest is an evergreen acceptance inventory. Dated assessment
packages remain immutable snapshots and may cite a specific corpus version and
commit.

## Verification Strategy

Implementation follows test-driven development:

1. failing schema/loader tests for the exact fifteen-scenario contract;
2. failing status and evidence-invariant tests;
3. failing registration/fixture parity tests;
4. failing production happy-path harness test;
5. failing file-backed close/reopen recovery test;
6. focused unit and integration suites;
7. repository formatting, type, and diff checks; and
8. requirement-by-requirement review against this design and the active goal.

## Non-goals

- Declaring the DAG complete.
- Implementing every currently missing authoring, contention, recovery, or
  scale capability in this package.
- Replacing the normative execution-graph contract or dated assessments.
- Treating documentation, direct graph tests, or skipped browser cases as
  executable production support.
- Adding another graph builder, recovery API, audit format, or CI-only bespoke
  enforcement script.
