# Token Outcome Assurance (TOA)

This document set defines how we guarantee that every token ends in the correct
terminal state and that the audit trail is complete, queryable, and defensible.

## Audience

- Engine developers
- QA / test owners
- Operators doing run validation and incident triage

## What this set covers

- Token outcome contract and required fields
- Outcome recording map (which code paths record which outcomes)
- Audit sweep checks (SQL) to find gaps fast
- Test strategy (unit, integration, property-based)
- Investigation playbook for gaps
- CI gates and quality metrics

## Quick start

1. Run the audit sweep after a completed run.
2. Classify any gaps by outcome type and path.
3. Use the outcome path map to locate the responsible code path.
4. Add a minimal regression test that reproduces the gap.

## Workstreams (delegation map)

- Workstream A: Contract + schema invariants (docs/audit-trail/tokens/00-token-outcome-contract.md)
- Workstream B: Outcome path mapping (docs/audit-trail/tokens/01-outcome-path-map.md)
- Workstream C: Audit sweep + SQL checks (docs/audit-trail/tokens/02-audit-sweep.md)
- Workstream D: Test strategy + coverage (docs/audit-trail/tokens/03-test-strategy.md)
- Workstream E: CI gates + metrics (docs/audit-trail/tokens/05-ci-gates-and-metrics.md)
- Workstream F: Investigation playbook (docs/audit-trail/tokens/04-investigation-playbook.md)

## Index

- 00-token-outcome-contract.md
- 01-outcome-path-map.md
- 02-audit-sweep.md
- 03-test-strategy.md
- 04-investigation-playbook.md
- 05-ci-gates-and-metrics.md
