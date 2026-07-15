# Task: Assess authoring, contracts, security, and maintainability

## Context

- Workspace: `docs/arch-analysis-2026-07-15-1415/`
- Read: `00-coordination.md` and `01-discovery-findings.md`.
- Baseline: commit `0dcd61acaa44082d93ec205683700e798748ee6d`.
- Write: `temp/evidence-authoring-contracts.md`.

## Scope

Assess freeform YAML, Composer import/export, guided authoring, canonical round-trip parity, contract documentation, scenario fixtures, skipped end-to-end tests, topology hashing/metadata secret handling, and graph/compiler maintainability risks.

Use Loomweave first for entity location and relationships. Verify claims against current source, tests, plans, and live Filigree state. Run focused tests when useful. Do not edit product code or existing documents.

## Expected output

- Surface-by-capability matrix for freeform, import, export, guided, and runtime build.
- Documentation drift findings with exact claims and replacement direction.
- Security and maintainability gaps that block a production-complete verdict.
- Prioritized acceptance fixtures and release gates.
- Existing Filigree issue IDs, with stale issues called out explicitly.
- Confidence and limitations.

## Validation criteria

- Verify queue support from live source before accepting tracker descriptions.
- Distinguish freeform support from guided authoring support.
- Treat skipped or planned parity suites as missing evidence.
- Flag raw-secret persistence or hashing risks as hard blockers.
