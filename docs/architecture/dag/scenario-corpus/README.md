# Maintained DAG Scenario Corpus

This directory holds the evergreen, executable inventory used to answer a
specific question: which parts of Elspeth's mandatory directed acyclic graph
(DAG) lifecycle have current production-path evidence?

Start with the [v1 manifest](v1/manifest.yaml). It contains all 15 mandatory
scenarios, all 11 assessment dimensions, the evidence registry, owned gaps,
observable exit gates, and the cases that the production-path harness runs.
The [DAG information hub](../README.md) supplies the broader completeness
assessment and remediation context.

## Authority boundary

These files have distinct jobs:

- The [completeness criteria](../completeness-criteria.md) define the quality
  bar and the mandatory scenario set. Change them only when the intended bar
  changes.
- The [v1 manifest](v1/manifest.yaml) is the authoritative live inventory of
  scenario cells, declared evidence, ownership, exit gates, and executable
  case declarations.
- The [typed schema](../../../../tests/fixtures/dag_scenario_corpus/schema.py)
  defines the closed manifest and observed-evidence shapes. It also derives
  `complete` only when every cell is `pass` or `not_applicable`.
- The [strict loader](../../../../tests/fixtures/dag_scenario_corpus/loader.py)
  binds the manifest to the exact scenario and dimension inventory, rejects
  duplicate or orphaned declarations, validates fixtures, and checks evidence
  locators.
- The [production-path harness](../../../../tests/fixtures/dag_scenario_corpus/harness.py)
  executes registered cases and returns one common `ScenarioRunEvidence`
  record for configuration, build, runtime, audit, and recovery facts.

The manifest does not replace the criteria, and a dated assessment does not
replace the live manifest. Documentary evidence can explain a cell, but only
executable `harness` or `pytest` evidence can support `pass`.

## Status vocabulary

The manifest accepts exactly these lower-case statuses:

| Status | Meaning | Required shape |
| --- | --- | --- |
| `pass` | Current executable evidence proves the complete requirement for this cell. | One or more evidence IDs, including at least one `harness` or `pytest` reference; no reason, owner, or exit gate. |
| `partial` | Current evidence proves part, but not all, of the requirement. | A precise reason, Filigree owner issue, and observable exit gate. Evidence may be attached. |
| `fail` | Current evidence demonstrates behavior that misses the requirement. | A precise reason, Filigree owner issue, and observable exit gate. Evidence may be attached. |
| `unknown` | Adequate current production-path evidence has not been executed or does not exist. | A precise reason, Filigree owner issue, and observable exit gate. Evidence may be attached. |
| `not_applicable` | The dimension genuinely does not apply to this scenario. | A narrow applicability reason; no evidence, owner, or exit gate. |

`unknown` is a result, not a skipped test and not permission to infer success.
Keep the cell visible and owned until executable evidence proves a different
status. Registered harness cases must run normally: do not hide a coverage gap
with `skip`, `xfail`, or a plan-only reference.

## Register executable evidence

Use one of the two executable evidence kinds.

For a corpus harness case:

1. Add deterministic inputs and canonical YAML below
   `tests/fixtures/dag_scenario_corpus/v1/<scenario-id>/`.
2. Add a case beneath that scenario's `cases` list. Its locator is
   `<scenario-id>:<case-id>`.
3. Add one top-level evidence record with `kind: harness`, the same locator,
   a precise claim, and the stages it proves.
4. Reference that evidence ID only from cells its assertions actually prove.
5. Extend the table-driven assertions in the
   [production-path integration test](../../../../tests/integration/core/dag/test_dag_scenario_production_path.py)
   when the common expectation schema is not sufficient.

For an existing executable test, add a top-level record with `kind: pytest`
and a repository-relative pytest node locator such as
`tests/path/test_file.py::test_name`. The loader validates that the file and
node exist, and the contract suite batch-collects every declared pytest
locator.

Use `document` and `decision` references only as supporting context. They
cannot make a cell pass by themselves.

## Promote a cell

Promote evidence and status in the same commit:

1. Add or strengthen the executable assertion and observe it fail for the
   missing behavior or proof.
2. Make the production path and assertion pass.
3. Register the exact evidence locator in the manifest.
4. Attach the evidence ID to every cell it directly proves.
5. Change a cell to `pass` only when that evidence covers the whole cell, then
   remove its `reason`, `owner_issue`, and `exit_gate` fields.
6. Run the focused contract and integration suites before committing.

If evidence closes only part of the gap, keep `partial` and rewrite its reason
and exit gate to state exactly what remains. Do not promote a nearby cell by
analogy.

## Run the focused checks

From the repository root, validate the manifest, schema, locators, fixtures,
documentation links, and evidence contracts:

```bash
.venv/bin/pytest -q \
  tests/unit/architecture/test_dag_scenario_corpus_contract.py
```

Run every registered production-path harness case:

```bash
.venv/bin/pytest -q \
  tests/integration/core/dag/test_dag_scenario_production_path.py
```

The [unit contract suite](../../../../tests/unit/architecture/test_dag_scenario_corpus_contract.py)
must reject malformed inventory or evidence. The integration suite must run
registered cases without skips or expected failures and must assert the
observed evidence, not merely successful process exit.

## Dated assessments

The live corpus evolves; [dated assessments](../assessments/) remain immutable
records of one commit. A new assessment should cite:

- `docs/architecture/dag/scenario-corpus/v1/manifest.yaml`;
- the manifest `schema_version`;
- the assessed Git commit; and
- the exact corpus commands and observed results.

Do not rewrite an older assessment when the manifest changes. Add a new dated
assessment, or add an explicit erratum when the older record itself is wrong.
The [assessment framework](../assessment-framework.md) defines the complete
snapshot workflow.

## Active Filigree work

The foundation and remaining corpus coverage are tracked by
Filigree issue `elspeth-ef29ef6ba4`. Inspect its live state rather than copying
a status into this evergreen page:

```bash
filigree show elspeth-ef29ef6ba4 --json
```

Keep the issue open while applicable cells still rely on incomplete evidence
owned by it. Close it only when its full acceptance scope—not merely the
manifest and harness foundation—is satisfied.
