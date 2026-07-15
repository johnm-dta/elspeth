# Validation Report: DAG Completeness Gap Analysis

**Validator status:** `NEEDS_REVISION (warnings)`
**Independent review basis:** all numbered assessment documents, all three evidence reports, the committed Wave 1 scheduler ledger, live tracker records, source consistency checks, link checks, and independent reruns of the 62-test DAG property suite, 66-test Composer suite, and 26-test scheduler/contention suite.

The validator completed the independent review but the shared process launcher exhausted file descriptors before it could persist this file. The coordinator transcribed the validator's returned findings below without changing their substance.

## Findings

### V-01 — Scenario passes exceed observed evidence

- **Document:** `04-dag-completeness-gap-analysis.md`, scenario gap matrix.
- **Evidence:** Runtime was marked `Pass` for conditional routing, fork/coalesce, aggregation, and row expansion, while the assessment's recorded executions cover DAG properties, Composer import/export, scheduler/contention, and two coalesce-to-gate tests. Several other production tests were inspected but not rerun.
- **Required change:** Downgrade unsupported `Pass` cells to `Partial`/`Unknown`, or add exact executed evidence.
- **Severity:** Warning.
- **Disposition:** Addressed. Runtime cells without an observed assessment run are now `Partial` or `Unknown`.

### V-02 — Security consequence is overstated

- **Document:** `04-dag-completeness-gap-analysis.md`, G6.
- **Evidence:** Raw metadata can disclose credentials directly; raw-config hashing creates an offline oracle or correlation surface. A hash does not itself expose raw credentials.
- **Required change:** Separate direct metadata disclosure from hash oracle/correlation risk and verify redacted/fingerprinted hash inputs.
- **Severity:** Warning.
- **Disposition:** Addressed. G6 now distinguishes the two consequences and strengthens the exit gate.

### V-03 — Scoring scale needs dimension-local clarification

- **Documents:** `02-capability-evidence.md`, `03-completeness-model.md`.
- **Evidence:** The global score-4 definition included recovery, contention, audit, and authoring, while topology and structural-validation dimensions scored 4 despite failures elsewhere.
- **Required change:** State that dimension scores use layer-local criteria or provide dimension-specific rubrics; preserve the hard-gate verdict.
- **Severity:** Warning.
- **Disposition:** Addressed. Both documents now define layer-local scores and make the weakest mandatory cells plus hard gates authoritative for the product verdict.

## Gate outcome

All validator warnings were accepted and corrected. A final documentation and diff verification pass is still required before release of the assessment.
