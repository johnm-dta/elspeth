# Task: Assess core DAG semantics

## Context

- Workspace: `docs/arch-analysis-2026-07-15-1415/`
- Read: `00-coordination.md` and `01-discovery-findings.md`.
- Baseline: commit `0dcd61acaa44082d93ec205683700e798748ee6d`.
- Write: `temp/evidence-core-dag.md`.

## Scope

Assess graph configuration, builder/compiler behavior, structural validation, schema propagation, cardinality, and compositional closure. Cover sources, transforms, gates, sinks, queues, coalesces, aggregations, row expansion, nested/sequential forks, and multi-source graphs.

Use Loomweave first for entity location and relationships. Verify claims against exact current source and tests. Run focused tests when useful. Do not edit product code or existing documents.

## Expected output

- Supported capability inventory with file/test evidence.
- Scenario verdicts using `Pass`, `Partial`, `Fail`, or `Unknown`.
- Concrete gaps with impact, evidence, and recommended acceptance test.
- Existing Filigree issue IDs where they genuinely match the live gap.
- Confidence and limitations.

## Validation criteria

- Distinguish modeled, compiled, executed, and maintained behavior.
- Do not treat a planned or skipped test as passing evidence.
- Identify at least one negative validation test for each claimed structural invariant.
- Flag any gap that threatens loss, duplication, replay safety, or secret handling.
