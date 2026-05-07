# ADR-016: Source Guaranteed Fields Contract

**Status:** Accepted
**Date:** 2026-04-20
**Supersedes:** none
**Depends on:** ADR-010

## Context

Source plugins can advertise stable producer guarantees through schema config
and, at runtime, through `declared_guaranteed_fields`. Downstream propagation,
DAG validation, and audit interpretation all rely on those guarantees being
true. Before this ADR, ELSPETH had no dispatcher-owned runtime check that a
source row actually satisfied its declared guarantees.

## Decision

Introduce `SourceGuaranteedFieldsContract` as a `boundary_check` adopter under
ADR-010.

- Violation class: `SourceGuaranteedFieldsViolation`
- Payload schema: `SourceGuaranteedFieldsPayload`
- Tier: 1
- Runtime observation: `row_contract.fields ∩ row_data.keys()`
- Call posture: run after token creation in `RowProcessor.process_row()`, never
  on `process_existing_row()`
- Failure recording: record a terminal `FAILED` token outcome plus a `FAILED`
  source node state before re-raising the Tier 1 exception

## Rationale

- Source guarantees are producer-side contract claims, not best-effort hints.
- Payload membership alone is insufficient. If the payload contains a key that
  the emitted row contract omits, downstream contract propagation still sees a
  lie.
- Resume runs must not re-cross the source boundary. Re-validating the original
  source output on resume would mint fresh source failures for historical data.

## Consequences

- Valid source rows that violate declared guarantees now fail loudly with
  row-level attribution.
- Quarantined source rows remain on the source-validation path only; they do
  not also run the boundary contract.
- Source plugins must expose `declared_guaranteed_fields` as a runtime
  attribute derived from effective schema config after any source-local schema
  rewrites.
