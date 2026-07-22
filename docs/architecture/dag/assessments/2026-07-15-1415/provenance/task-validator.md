# Task: Independently validate the DAG completeness analysis

## Context

- Workspace: `docs/arch-analysis-2026-07-15-1415/`
- Baseline: `release/0.7.1` at `0dcd61acaa44082d93ec205683700e798748ee6d`.
- Read all numbered workspace documents and all three `temp/evidence-*.md` reports.
- Write only `temp/validation-04-final-report.md`.

## Validation contract

Check the final report against current source, tests, live tracker evidence already cited, and its own completeness model.

Validate:

- every high-impact claim has exact evidence;
- reproduced defects are distinct from unknown/unproved seams;
- stale tracker narratives are not repeated as current facts;
- scores are internally consistent and do not average away hard gates;
- scenario verdicts match the detailed evidence;
- priority and sequencing follow risk (loss, duplication, stale-worker writes, audit divergence, secret exposure);
- proposed exit gates are observable and testable;
- links, headings, Mermaid, tables, and relative paths are valid;
- the report does not claim tests that were not observed;
- the report does not modify or contradict the committed Wave 1 scheduler evidence.

## Output

Return one status:

- `APPROVED`;
- `NEEDS_REVISION (warnings)` with specific non-blocking fixes; or
- `NEEDS_REVISION (critical)` with exact blockers.

For every finding include document, section/line, evidence, required change, and severity. Include a validation evidence section listing commands or source checks actually run.

Do not edit the report or any product file. Do not update Filigree.
