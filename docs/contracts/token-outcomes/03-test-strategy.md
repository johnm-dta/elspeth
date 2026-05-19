# Token Outcome Test Strategy

Current as of 2026-05-20.

Tests should prove the two-axis token outcome contract, not the retired
single-axis `RowOutcome` model.

## Invariants

1. Every token has exactly one completed outcome after terminal run completion.
2. Non-terminal rows are limited to `(outcome IS NULL, path='buffered',
   completed=0)`.
3. Completed rows use a legal `(outcome, path)` pair.
4. Pair-specific required fields are present.
5. Pair-specific forbidden fields are absent.
6. Completed sink states and success outcomes with sinks agree.
7. Fork, expand, coalesce, and batch-consumed paths preserve parent/child or
   batch evidence.

## Unit Tests

Use unit tests for producer-specific rules:

- `TerminalOutcome` / `TerminalPath` legal-pair coverage in
  `src/elspeth/contracts/enums.py`.
- `TokenOutcome` field constraints in `src/elspeth/contracts/audit.py`.
- Data-flow repository writes for parent, child, batch, and buffered outcomes.
- Processor branches for gate, transform-error, filter/drop, batch, and
  coalesce paths.
- Sink executor branches for normal success, failsink fallback, and discard.

## Integration Tests

Use minimal pipelines that exercise the full config-to-runtime path:

- Default source -> transform -> sink success.
- Gate route to named sink and gate discard.
- Transform `on_error` to sink and discard/failure paths.
- Filter/drop path.
- Fork plus coalesce.
- Batch buffer, batch consumed, and flush.
- Source quarantine.

After each run, execute the relevant sweep queries from
[Audit Sweep](02-audit-sweep.md).

## Property Tests

Use Hypothesis for invariant hunting:

- Generate small DAG paths with fork, expand, batch, and coalesce operations.
- Assert legal pair coverage and required/forbidden fields.
- Assert parent/delegation paths produce corresponding child or batch evidence.
- Keep PR profiles bounded; use wider profiles in nightly or explicit audit
  sweeps.

## CI Scheduling

- PR: focused unit tests, representative integration tests, bounded property
  profile.
- Main: broader unit and integration test sweep.
- Nightly or release gate: expanded property profile and end-to-end examples
  with audit sweep checks.
