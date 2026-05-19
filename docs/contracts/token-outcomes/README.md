# Token Outcome Assurance

Current as of 2026-05-20.

This document set defines how ELSPETH proves that token lifecycles are complete,
queryable, and defensible in Landscape.

The current model is ADR-019's two-axis terminal model:

- `completed` answers whether this `token_outcomes` row is terminal.
- `outcome` answers the lifecycle result: `success`, `failure`, `transient`, or
  `NULL` for a buffered non-terminal row.
- `path` answers how the producer says the token reached that result.

Authoritative source references:

- `src/elspeth/contracts/enums.py` - `TerminalOutcome`, `TerminalPath`, legal pair coverage.
- `src/elspeth/contracts/audit.py` - `TokenOutcome` and field constraints.
- `src/elspeth/core/landscape/schema.py` - `token_outcomes` table and partial terminal index.
- `src/elspeth/core/landscape/data_flow_repository.py` - token outcome write paths.

## Audience

- Engine developers
- QA and test owners
- Operators doing run validation and incident triage

## Index

- [Token outcome contract](00-token-outcome-contract.md)
- [Outcome path map](01-outcome-path-map.md)
- [Audit sweep](02-audit-sweep.md)
- [Test strategy](03-test-strategy.md)
- [Investigation playbook](04-investigation-playbook.md)
- [CI gates and metrics](05-ci-gates-and-metrics.md)

## Quick Start

1. Run the audit sweep after a run is terminal.
2. Classify any gap by `(outcome, path, completed)`.
3. Use the outcome path map to locate the producer.
4. Add the smallest regression that reproduces the gap.
5. Re-run the sweep and the focused regression.
